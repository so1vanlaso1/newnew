"""Convert the dataset's FOL formulas (Unicode AND Pythonic flavors) into Z3 Python DSL.

Input examples we have to handle:

    ∀x (WT(x) → O(x))
    ∀x (¬PEP8(x) → ¬WT(x))
    ∃x (S(x) ∧ E(x))
    ForAll(x, E(x) → U(x))
    ForAll(s, ForAll(m, (attendance(s,m) ≥ 80) → allowed_exam(s,m)))
    Exists(x, Professor(x) ∧ Concern(x))
    CreatesClass(John, Subject)
    AccessibleByInheritedClasses(Math) ∧ ¬AccessibleOutsideClass(Math)
    ∀x P(x) → ∀x (R(x) → S(x))
    (∀x (R(x) → S(x))) → (∃x A(x) → ∀x (¬E(x) → ¬R(x)))
    grade(s,m1) > 8.5
    m1 ≠ m2

Output is a list of Python statements + a top-level expression, e.g.:

    U = DeclareSort('U')
    WT = Function('WT', U, BoolSort())
    O = Function('O', U, BoolSort())
    x = Const('x', U)
    # expression:
    ForAll([x], Implies(WT(x), O(x)))

The renderer returns (setup_lines, expression). Multiple formulas sharing
the same predicate symbols are batched in `convert_premises_to_z3py`, which
deduplicates declarations.

Failure modes are explicit: if a formula uses something the parser doesn't
support yet (e.g. higher-order quantification, set-builder notation), the
parser raises `FolParseError` and the caller skips the record.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Iterable


class FolParseError(ValueError):
    pass


# Python reserved words can appear as predicate/constant names in the dataset
# FOL (e.g. a predicate literally named `pass`). Used as a bare Python
# identifier they are a SyntaxError, so we rename them to a safe form. The
# mapping is deterministic per-name, so declarations and uses stay consistent.
_RESERVED = set(keyword.kwlist) | set(getattr(keyword, "softkwlist", []))

# Z3 DSL constructor names injected into the exec sandbox (mirrors the allow-list
# in solver.z3_runner). A dataset symbol that happens to share one of these names
# (e.g. a predicate literally named `Function`, `Int`, `And`) would, once declared
# as `Function = Function('Function', …)`, shadow the constructor and break every
# later declaration/use. We rename such symbols to a safe form, consistently in
# both declarations and expressions.
_RESERVED_DSL = {
    "DeclareSort", "BoolSort", "IntSort", "RealSort", "Function", "Const",
    "Consts", "Bool", "Bools", "Int", "Ints", "Real", "Reals", "BoolVal",
    "IntVal", "RealVal", "Not", "And", "Or", "Implies", "Iff", "Xor",
    "ForAll", "Exists", "Distinct",
}


def safe_ident(name: str) -> str:
    """Map a FOL symbol to a valid Python identifier that is neither a Python
    keyword nor a Z3 DSL constructor name."""
    if name in _RESERVED or name in _RESERVED_DSL:
        return name + "_"
    return name


# ─────────────────────────────────────────────────────────────────────────
# AST
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Node:
    pass


@dataclass
class Var(Node):
    name: str


@dataclass
class Const(Node):
    name: str


@dataclass
class Num(Node):
    value: float
    is_int: bool


@dataclass
class App(Node):
    name: str
    args: list[Node]


@dataclass
class Not(Node):
    body: Node


@dataclass
class BinOp(Node):
    op: str  # 'and', 'or', 'implies', 'iff'
    left: Node
    right: Node


@dataclass
class Cmp(Node):
    op: str  # '=', '!=', '<', '>', '<=', '>='
    left: Node
    right: Node


@dataclass
class Arith(Node):
    op: str  # '+', '-', '*', '/'
    left: Node
    right: Node


@dataclass
class Quant(Node):
    kind: str  # 'forall', 'exists'
    vars: list[str]
    body: Node


# ─────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────


# Multi-char tokens first so the regex picks them before single-char ones.
_TOKEN_SPEC: list[tuple[str, str]] = [
    ("WS", r"[ \t\n\r]+"),
    ("NUMBER", r"\d+(?:\.\d+)?"),
    ("FORALL", r"∀|ForAll|Forall|forall|For_all|FORALL"),
    ("EXISTS", r"∃|Exists|exists|EXISTS"),
    ("NOT", r"¬|~|\bnot\b"),
    ("STRING", r"'[^']*'|\"[^\"]*\""),
    ("IFF", r"↔|<->|<=>"),
    ("IMPLIES", r"→|->|=>|\bimplies\b"),
    ("AND", r"∧|&&|/\\"),
    ("OR", r"∨|\|\||\\/"),
    ("NEQ", r"≠|!="),
    ("LEQ", r"≤|<="),
    ("GEQ", r"≥|>="),
    ("LT", r"<"),
    ("GT", r">"),
    ("EQ", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("COMMA", r","),
    ("DOT", r"\."),
    ("IDENT", r"[A-Za-z_][A-Za-z_0-9]*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKEN_SPEC))


@dataclass
class Token:
    kind: str
    value: str
    pos: int


def tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise FolParseError(f"unexpected char {src[pos]!r} at {pos}")
        kind = m.lastgroup or ""
        if kind != "WS":
            tokens.append(Token(kind, m.group(), pos))
        pos = m.end()
    return tokens


# ─────────────────────────────────────────────────────────────────────────
# Parser (recursive descent, Pratt-style for precedence)
# ─────────────────────────────────────────────────────────────────────────


class Parser:
    """Grammar (precedence low → high):

        formula = iff
        iff     = implies (IFF implies)*
        implies = orx    (IMPLIES orx)*          (right-associative)
        orx     = andx   (OR andx)*
        andx    = unary  (AND unary)*
        unary   = NOT unary | quant | atom
        quant   = (FORALL|EXISTS) ('[' var (',' var)* ']' | var+) ('.'|',')? unary
                | (FORALL|EXISTS) '(' var (',' var)* ',' formula ')'
        atom    = comparison | predicate | '(' formula ')'
        comparison = term cmp_op term
        term    = atom_term ((PLUS|MINUS|MUL|DIV) atom_term)*
        atom_term = NUMBER | IDENT '(' term (',' term)* ')' | IDENT | '(' term ')' | -term
        predicate = IDENT '(' term (',' term)* ')' | IDENT     -- nullary
    """

    KEYWORDS = {"And", "Or", "Not", "Implies", "Iff", "ForAll", "Exists"}

    def __init__(self, tokens: list[Token], src: str):
        self.tokens = tokens
        self.pos = 0
        self.src = src

    def peek(self, offset: int = 0) -> Token | None:
        i = self.pos + offset
        return self.tokens[i] if i < len(self.tokens) else None

    def eat(self, kind: str) -> Token:
        t = self.peek()
        if t is None or t.kind != kind:
            raise FolParseError(
                f"expected {kind}, got {t.kind if t else 'EOF'} ({t.value if t else ''!r}) "
                f"in: {self.src!r}"
            )
        self.pos += 1
        return t

    def accept(self, *kinds: str) -> Token | None:
        t = self.peek()
        if t and t.kind in kinds:
            self.pos += 1
            return t
        return None

    # ── Top-level ────────────────────────────────────────────────────────

    def parse_formula(self) -> Node:
        return self.parse_iff()

    def parse_iff(self) -> Node:
        left = self.parse_implies()
        while self.accept("IFF"):
            right = self.parse_implies()
            left = BinOp("iff", left, right)
        return left

    def parse_implies(self) -> Node:
        # Right-associative.
        left = self.parse_or()
        if self.accept("IMPLIES"):
            right = self.parse_implies()
            return BinOp("implies", left, right)
        return left

    def parse_or(self) -> Node:
        left = self.parse_and()
        while self.accept("OR"):
            right = self.parse_and()
            left = BinOp("or", left, right)
        return left

    def parse_and(self) -> Node:
        left = self.parse_unary()
        while self.accept("AND"):
            right = self.parse_unary()
            left = BinOp("and", left, right)
        return left

    def parse_unary(self) -> Node:
        if self.accept("NOT"):
            return Not(self.parse_unary())
        if self.peek() and self.peek().kind in ("FORALL", "EXISTS"):
            return self.parse_quant()
        # Function-call style And/Or/Not/Implies/Iff(…)
        t = self.peek()
        if t and t.kind == "IDENT" and t.value in self.KEYWORDS and self._look_ahead_lparen():
            return self.parse_keyword_call()
        return self.parse_atom()

    def _look_ahead_lparen(self) -> bool:
        nxt = self.peek(1)
        return nxt is not None and nxt.kind == "LPAREN"

    def parse_keyword_call(self) -> Node:
        kw = self.eat("IDENT").value
        self.eat("LPAREN")
        args: list[Node] = []
        if not self.accept("RPAREN"):
            args.append(self.parse_formula())
            while self.accept("COMMA"):
                args.append(self.parse_formula())
            self.eat("RPAREN")
        if kw == "Not":
            if len(args) != 1:
                raise FolParseError(f"Not() expects 1 arg, got {len(args)}")
            return Not(args[0])
        if kw == "And":
            return self._fold_binop("and", args)
        if kw == "Or":
            return self._fold_binop("or", args)
        if kw == "Implies":
            if len(args) != 2:
                raise FolParseError(f"Implies expects 2 args, got {len(args)}")
            return BinOp("implies", args[0], args[1])
        if kw == "Iff":
            if len(args) != 2:
                raise FolParseError(f"Iff expects 2 args, got {len(args)}")
            return BinOp("iff", args[0], args[1])
        raise FolParseError(f"unexpected keyword {kw}")

    def _fold_binop(self, op: str, args: list[Node]) -> Node:
        if not args:
            raise FolParseError(f"empty {op}")
        if len(args) == 1:
            return args[0]
        out = args[0]
        for a in args[1:]:
            out = BinOp(op, out, a)
        return out

    # ── Quantifiers ──────────────────────────────────────────────────────

    def parse_quant(self) -> Node:
        t = self.eat(self.peek().kind)  # FORALL or EXISTS
        kind = "forall" if t.kind == "FORALL" else "exists"

        # Pythonic style: ForAll(x, body)  or  ForAll([x, y], body)
        if self.accept("LPAREN"):
            vars_: list[str] = []
            if self.accept("LBRACK"):
                vars_.append(self._eat_var_ident())
                while self.accept("COMMA"):
                    vars_.append(self._eat_var_ident())
                self.eat("RBRACK")
            else:
                vars_.append(self._eat_var_ident())
                # Allow ForAll(x1, x2, x3, body) — last is body.
                # We accumulate until we see a comma followed by a non-IDENT or a body.
                # Simpler: keep eating ident-then-comma greedily, then parse body.
                while self.accept("COMMA"):
                    # peek: is next another bare ident followed by comma? if so, it's a var.
                    t2 = self.peek()
                    t3 = self.peek(1)
                    if (
                        t2 is not None
                        and t2.kind == "IDENT"
                        and t3 is not None
                        and t3.kind == "COMMA"
                    ):
                        vars_.append(self._eat_var_ident())
                        continue
                    # Otherwise the next thing IS the body.
                    body = self.parse_formula()
                    self.eat("RPAREN")
                    return self._build_quant(kind, vars_, body)
                # If we got here, no comma — single arg form, error.
                raise FolParseError(f"{kind} expects ',' before body")
            self.eat("COMMA")
            body = self.parse_formula()
            self.eat("RPAREN")
            return self._build_quant(kind, vars_, body)

        # Unicode style: ∀x BODY  or  ∀x y z BODY  or  ∀x (BODY)
        vars_ = [self._eat_var_ident()]
        # Allow chained quantifiers: ∀x ∀y or ∀x y
        while True:
            nt = self.peek()
            if nt is None:
                break
            if nt.kind in ("FORALL", "EXISTS"):
                # Defer: outer quantifier ends here; the next one parses recursively.
                body = self.parse_unary()
                return self._build_quant(kind, vars_, body)
            if nt.kind == "IDENT" and self._is_bare_var_chain(nt):
                # Heuristic: a bare ident with no following '(' or operator is another var.
                vars_.append(self._eat_var_ident())
                continue
            break
        # Optional separator after vars: `.` or `,`. The separator decides the
        # quantifier's scope, matching how the dataset writes these:
        #   * WITH a separator (`∃d, has_degree(x,d) ∧ higher(d,BSc)`) the form
        #     mirrors the Pythonic `Exists(d, body)` where the body is the whole
        #     following formula, so the scope extends maximally. Without this,
        #     only `has_degree(x,d)` would bind and `d` leaks free into
        #     `higher(d,BSc)`.
        #   * WITHOUT a separator (`∃x R(x) → ¬∃x S(x)`, or `∃d (φ) → ψ`) the
        #     quantifier binds just the immediately following atom / parenthesised
        #     group; the `→` stays at the outer level. So the body is a single
        #     `unary`, which already stops at a complete parenthesised group.
        separator = self.accept("DOT") or self.accept("COMMA")
        body = self.parse_formula() if separator else self.parse_unary()
        return self._build_quant(kind, vars_, body)

    def _is_bare_var_chain(self, t: Token) -> bool:
        """Heuristic: token at peek() is a single-letter ident with no `(` after."""
        if len(t.value) > 2:
            return False
        nxt = self.peek(1)
        return nxt is None or nxt.kind not in ("LPAREN", "LBRACK")

    def _build_quant(self, kind: str, vars_: list[str], body: Node) -> Node:
        # Pull off nested same-kind quantifiers into a single binder when convenient.
        if isinstance(body, Quant) and body.kind == kind:
            return Quant(kind, vars_ + body.vars, body.body)
        return Quant(kind, vars_, body)

    def _eat_var_ident(self) -> str:
        t = self.eat("IDENT")
        return t.value

    # ── Atoms / terms ────────────────────────────────────────────────────

    def parse_atom(self) -> Node:
        # Try comparison; if next sequence is a term-then-cmp-then-term, treat as Cmp.
        save = self.pos
        try:
            left = self.parse_term()
            t = self.peek()
            if t and t.kind in ("EQ", "NEQ", "LT", "GT", "LEQ", "GEQ"):
                op_map = {"EQ": "==", "NEQ": "!=", "LT": "<", "GT": ">", "LEQ": "<=", "GEQ": ">="}
                op = op_map[t.kind]
                self.pos += 1
                right = self.parse_term()
                return Cmp(op, left, right)
            # No comparison ⇒ the term itself must be a Boolean-valued atom (predicate
            # application or nullary predicate). A bare identifier (case-insensitive)
            # is promoted to a nullary predicate — signature collection elsewhere
            # already handles the bound-var-vs-nullary disambiguation.
            if isinstance(left, App):
                return left
            if isinstance(left, (Const, Var)):
                return App(left.name, [])
            raise FolParseError(f"bare term in boolean position: {left}")
        except FolParseError:
            self.pos = save
            # Maybe a parenthesized formula.
            if self.accept("LPAREN"):
                body = self.parse_formula()
                self.eat("RPAREN")
                return body
            raise

    def parse_term(self) -> Node:
        return self._parse_term_addsub()

    def _parse_term_addsub(self) -> Node:
        left = self._parse_term_muldiv()
        while True:
            t = self.peek()
            if t and t.kind in ("PLUS", "MINUS"):
                op = "+" if t.kind == "PLUS" else "-"
                self.pos += 1
                right = self._parse_term_muldiv()
                left = Arith(op, left, right)
            else:
                return left

    def _parse_term_muldiv(self) -> Node:
        left = self._parse_term_unary()
        while True:
            t = self.peek()
            if t and t.kind in ("MUL", "DIV"):
                op = "*" if t.kind == "MUL" else "/"
                self.pos += 1
                right = self._parse_term_unary()
                left = Arith(op, left, right)
            else:
                return left

    def _parse_term_unary(self) -> Node:
        if self.accept("MINUS"):
            inner = self._parse_term_unary()
            return Arith("-", Num(0, True), inner)
        return self._parse_term_atom()

    def _parse_term_atom(self) -> Node:
        t = self.peek()
        if t is None:
            raise FolParseError("unexpected EOF in term")
        if t.kind == "NUMBER":
            self.pos += 1
            is_int = "." not in t.value
            return Num(float(t.value), is_int)
        if t.kind == "STRING":
            # 'PoliticalIdeologies' → constant str_PoliticalIdeologies (sanitized).
            self.pos += 1
            raw = t.value[1:-1]
            safe = re.sub(r"[^A-Za-z0-9_]", "_", raw)
            return Const(f"str_{safe}")
        if t.kind == "LPAREN":
            self.pos += 1
            inner = self._parse_term_addsub()
            self.eat("RPAREN")
            return inner
        if t.kind == "IDENT":
            name = t.value
            self.pos += 1
            if self.accept("LPAREN"):
                args: list[Node] = []
                if not self.accept("RPAREN"):
                    args.append(self._parse_term_addsub())
                    while self.accept("COMMA"):
                        args.append(self._parse_term_addsub())
                    self.eat("RPAREN")
                return App(name, args)
            # Heuristic: lowercase identifier ⇒ variable; capitalized ⇒ constant.
            if name and name[0].islower():
                return Var(name)
            return Const(name)
        raise FolParseError(f"unexpected token {t.kind} {t.value!r} in term")


def parse(src: str) -> Node:
    src = src.strip()
    if not src:
        raise FolParseError("empty input")
    tokens = tokenize(src)
    p = Parser(tokens, src)
    node = p.parse_formula()
    if p.pos < len(tokens):
        raise FolParseError(
            f"trailing tokens after parse: {[t.value for t in tokens[p.pos:]]!r}"
        )
    return node


# ─────────────────────────────────────────────────────────────────────────
# Signature collection (to emit Z3 declarations)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Signature:
    # `U` collides with common single-letter predicate names in the dataset
    # (rec 0 already has a predicate named `U`). `Universe` is unambiguous.
    sort_name: str = "Universe"
    # name → arity (predicates always Bool-valued; arithmetic functions handled separately)
    predicates: dict[str, int] = field(default_factory=dict)
    # name → arity (returns RealSort; used for numeric terms like attendance(s,m))
    int_functions: dict[str, int] = field(default_factory=dict)
    # name → arity (returns the universe sort; used for non-numeric *term*-position
    # applications such as `sides(ABC)` inside `proportional(sides(ABC), …)` — these
    # denote universe elements, not Booleans, so they must NOT be declared as predicates.
    universe_functions: dict[str, int] = field(default_factory=dict)
    # set of bare constants
    constants: set[str] = field(default_factory=set)
    # set of bound variable names (must NOT be declared as constants)
    bound_vars: set[str] = field(default_factory=set)
    # constants / bound vars that must be declared RealSort (used numerically,
    # e.g. `h >= 500` or passed into a numeric function-argument position).
    numeric_names: set[str] = field(default_factory=set)
    # function/predicate name → set of argument indices that take a numeric
    # (RealSort) value rather than the universe sort, e.g. `clinical_hours(s, 600)`.
    func_arg_real: dict[str, set[int]] = field(default_factory=dict)

    def merge(self, other: "Signature") -> None:
        for k, v in other.predicates.items():
            if k in self.predicates and self.predicates[k] != v:
                raise FolParseError(
                    f"predicate {k} used with arity {self.predicates[k]} and {v}"
                )
            self.predicates[k] = v
        for k, v in other.int_functions.items():
            if k in self.int_functions and self.int_functions[k] != v:
                raise FolParseError(
                    f"function {k} used with arity {self.int_functions[k]} and {v}"
                )
            self.int_functions[k] = v
        for k, v in other.universe_functions.items():
            if k in self.universe_functions and self.universe_functions[k] != v:
                raise FolParseError(
                    f"function {k} used with arity {self.universe_functions[k]} and {v}"
                )
            self.universe_functions[k] = v
        self.constants |= other.constants
        self.bound_vars |= other.bound_vars
        self.numeric_names |= other.numeric_names
        for k, v in other.func_arg_real.items():
            self.func_arg_real.setdefault(k, set()).update(v)


def _collect_in_term(node: Node, sig: Signature, locals_: set[str]) -> bool:
    """Walk a term subtree, recording signatures. Returns True if any sub-term
    is "arithmetic-flavored" (number, comparison input, function returning Int).
    """
    if isinstance(node, Var):
        return False
    if isinstance(node, Const):
        if node.name not in locals_:
            sig.constants.add(node.name)
        return False
    if isinstance(node, Num):
        return True
    if isinstance(node, App):
        arity = len(node.args)
        any_arith = any(_collect_in_term(a, sig, locals_) for a in node.args)
        # This App sits in *term* position (it is an argument to another
        # application, or an operand of a comparison/arithmetic op). It therefore
        # denotes a value, not a Boolean. If any argument is arithmetic-flavored we
        # treat it as a numeric (RealSort) function; otherwise it returns a universe
        # element. Either way it must NOT be declared as a Bool predicate, or Z3
        # raises a sort mismatch when the value is fed into the enclosing term.
        if any_arith:
            sig.int_functions[node.name] = arity
        else:
            sig.universe_functions.setdefault(node.name, arity)
        return any_arith
    if isinstance(node, Arith):
        l = _collect_in_term(node.left, sig, locals_)
        r = _collect_in_term(node.right, sig, locals_)
        return True
    if isinstance(node, Cmp):
        _collect_in_term(node.left, sig, locals_)
        _collect_in_term(node.right, sig, locals_)
        return False
    raise FolParseError(f"term position holds non-term: {node}")


def collect_signature(node: Node, sig: Signature | None = None,
                      locals_: set[str] | None = None) -> Signature:
    sig = sig or Signature()
    locals_ = locals_ or set()
    if isinstance(node, Quant):
        new_locals = locals_ | set(node.vars)
        sig.bound_vars |= set(node.vars)
        collect_signature(node.body, sig, new_locals)
        return sig
    if isinstance(node, Not):
        collect_signature(node.body, sig, locals_)
        return sig
    if isinstance(node, BinOp):
        collect_signature(node.left, sig, locals_)
        collect_signature(node.right, sig, locals_)
        return sig
    if isinstance(node, Cmp):
        # Both sides are Int-valued terms. Upgrade any App on either side
        # from predicate-default to int_function.
        for side in (node.left, node.right):
            _promote_apps_to_int(side, sig, locals_)
        return sig
    if isinstance(node, App):
        # Predicate position.
        arity = len(node.args)
        existing = sig.predicates.get(node.name)
        if existing is None:
            sig.predicates[node.name] = arity
        for a in node.args:
            _collect_in_term(a, sig, locals_)
        return sig
    if isinstance(node, Const):
        if node.name not in locals_:
            sig.constants.add(node.name)
        return sig
    if isinstance(node, Arith):
        # Top-level Arith in boolean position is meaningless; let caller deal.
        return sig
    raise FolParseError(f"unsupported node: {node}")


def _promote_apps_to_int(node: Node, sig: Signature, locals_: set[str]) -> None:
    if isinstance(node, App):
        arity = len(node.args)
        sig.int_functions[node.name] = arity
        sig.predicates.pop(node.name, None)
        sig.universe_functions.pop(node.name, None)
        for a in node.args:
            _collect_in_term(a, sig, locals_)
        return
    if isinstance(node, Arith):
        _promote_apps_to_int(node.left, sig, locals_)
        _promote_apps_to_int(node.right, sig, locals_)
        return
    if isinstance(node, (Const, Var, Num)):
        _collect_in_term(node, sig, locals_)
        return


# ─────────────────────────────────────────────────────────────────────────
# Renderer → Z3 Python DSL
# ─────────────────────────────────────────────────────────────────────────


def render_expr(node: Node, sig: Signature) -> str:
    if isinstance(node, Var):
        return safe_ident(node.name)
    if isinstance(node, Const):
        return safe_ident(node.name)
    if isinstance(node, Num):
        return repr(int(node.value)) if node.is_int else repr(node.value)
    if isinstance(node, App):
        if node.args:
            return f"{safe_ident(node.name)}(" + ", ".join(render_expr(a, sig) for a in node.args) + ")"
        return safe_ident(node.name)
    if isinstance(node, Not):
        return f"Not({render_expr(node.body, sig)})"
    if isinstance(node, BinOp):
        opmap = {"and": "And", "or": "Or", "implies": "Implies", "iff": "Iff"}
        return f"{opmap[node.op]}({render_expr(node.left, sig)}, {render_expr(node.right, sig)})"
    if isinstance(node, Cmp):
        return f"({render_expr(node.left, sig)} {node.op} {render_expr(node.right, sig)})"
    if isinstance(node, Arith):
        return f"({render_expr(node.left, sig)} {node.op} {render_expr(node.right, sig)})"
    if isinstance(node, Quant):
        kind = "ForAll" if node.kind == "forall" else "Exists"
        vars_ = ", ".join(safe_ident(v) for v in node.vars)
        return f"{kind}([{vars_}], {render_expr(node.body, sig)})"
    raise FolParseError(f"unrenderable node: {node}")


def render_setup(sig: Signature) -> list[str]:
    """Emit Z3 Python declarations for the collected signature."""
    lines: list[str] = []
    sort = sig.sort_name
    lines.append(f"{sort} = DeclareSort('{sort}')")

    def arg_sort_list(name: str, arity: int) -> str:
        reals = sig.func_arg_real.get(name, set())
        return ", ".join("RealSort()" if i in reals else sort for i in range(arity))

    # Predicates: Function(name, <arg sorts>, BoolSort())
    for name, arity in sorted(sig.predicates.items()):
        if arity == 0:
            lines.append(f"{safe_ident(name)} = Const('{name}', BoolSort())")
        else:
            lines.append(
                f"{safe_ident(name)} = Function('{name}', {arg_sort_list(name, arity)}, BoolSort())"
            )
    # Real-valued functions (handles `grade(s,m) > 8.5` and numeric arguments).
    for name, arity in sorted(sig.int_functions.items()):
        if arity == 0:
            lines.append(f"{safe_ident(name)} = Const('{name}', RealSort())")
        else:
            lines.append(
                f"{safe_ident(name)} = Function('{name}', {arg_sort_list(name, arity)}, RealSort())"
            )
    # Universe-valued functions (term-position applications like `sides(ABC)`).
    for name, arity in sorted(sig.universe_functions.items()):
        if arity == 0:
            lines.append(f"{safe_ident(name)} = Const('{name}', {sort})")
        else:
            lines.append(
                f"{safe_ident(name)} = Function('{name}', {arg_sort_list(name, arity)}, {sort})"
            )
    # Bound quantifier variables also need a `Const` declaration so that
    # `ForAll([x], …)` can reference `x`. Constants and bound vars are declared
    # the same way (both are Z3 Const objects in the namespace). A name used
    # numerically is declared RealSort so arithmetic/comparisons type-check.
    decl_names = sorted(sig.constants | sig.bound_vars)
    for c in decl_names:
        csort = "RealSort()" if c in sig.numeric_names else sort
        lines.append(f"{safe_ident(c)} = Const('{c}', {csort})")
    return lines


def collect_free_vars(node: Node, bound: set[str] | None = None) -> set[str]:
    """Find variables that are referenced but never bound by a quantifier."""
    bound = bound or set()
    if isinstance(node, Var):
        return set() if node.name in bound else {node.name}
    if isinstance(node, (Const, Num)):
        return set()
    if isinstance(node, Quant):
        return collect_free_vars(node.body, bound | set(node.vars))
    if isinstance(node, (Not,)):
        return collect_free_vars(node.body, bound)
    if isinstance(node, (BinOp, Cmp, Arith)):
        return collect_free_vars(node.left, bound) | collect_free_vars(node.right, bound)
    if isinstance(node, App):
        out: set[str] = set()
        for a in node.args:
            out |= collect_free_vars(a, bound)
        return out
    return set()


# ─────────────────────────────────────────────────────────────────────────
# Numeric-sort inference
# ─────────────────────────────────────────────────────────────────────────

_ORDER_OPS = {"<", ">", "<=", ">="}


def _is_numeric_term(node: Node, sig: Signature) -> bool:
    """True if `node` denotes a number (RealSort) rather than a universe element."""
    if isinstance(node, Num):
        return True
    if isinstance(node, Arith):
        return True
    if isinstance(node, App):
        return node.name in sig.int_functions
    if isinstance(node, (Const, Var)):
        return node.name in sig.numeric_names
    return False


def _infer_numeric_walk(node: Node, sig: Signature) -> bool:
    """One pass of numeric-sort propagation. Returns True if anything changed.

    Rules:
      * an order comparison (`<`,`>`,`<=`,`>=`) forces both operands numeric;
      * an arithmetic operand is numeric;
      * `==`/`!=` only propagate numeric-ness when one side is already numeric
        (so plain universe equality like `m1 != m2` stays universe-sorted);
      * a numeric function argument marks that arg position RealSort, and a
        RealSort arg position marks any constant/variable passed there numeric.
    """
    changed = False

    def mark_numeric(n: Node) -> None:
        nonlocal changed
        if isinstance(n, (Const, Var)) and n.name not in sig.numeric_names:
            sig.numeric_names.add(n.name)
            changed = True

    if isinstance(node, App):
        reals = sig.func_arg_real.setdefault(node.name, set())
        for i, a in enumerate(node.args):
            if _is_numeric_term(a, sig) and i not in reals:
                reals.add(i)
                changed = True
            if i in reals:
                mark_numeric(a)
            if _infer_numeric_walk(a, sig):
                changed = True
        return changed

    if isinstance(node, Cmp):
        numeric = (
            node.op in _ORDER_OPS
            or _is_numeric_term(node.left, sig)
            or _is_numeric_term(node.right, sig)
        )
        for side in (node.left, node.right):
            if numeric:
                mark_numeric(side)
            if _infer_numeric_walk(side, sig):
                changed = True
        return changed

    if isinstance(node, Arith):
        for side in (node.left, node.right):
            mark_numeric(side)
            if _infer_numeric_walk(side, sig):
                changed = True
        return changed

    if isinstance(node, Not):
        return _infer_numeric_walk(node.body, sig)
    if isinstance(node, Quant):
        return _infer_numeric_walk(node.body, sig)
    if isinstance(node, BinOp):
        left = _infer_numeric_walk(node.left, sig)
        right = _infer_numeric_walk(node.right, sig)
        return left or right
    return changed


def infer_numeric_sorts(nodes: Iterable[Node], sig: Signature) -> None:
    """Fixed-point numeric-sort inference over all parsed formulas."""
    node_list = [n for n in nodes if n is not None]
    changed = True
    while changed:
        changed = False
        for n in node_list:
            if _infer_numeric_walk(n, sig):
                changed = True


# ─────────────────────────────────────────────────────────────────────────
# Symbol-usage analysis & conflict resolution
# ─────────────────────────────────────────────────────────────────────────
#
# The source FOL is frequently inconsistent across formulas in a single record:
# the same symbol is used as both a predicate and an individual constant
# (`Bob(x)` in one premise, `StudyHard(Bob)` in another), or as a function of
# different arities, or as both a Boolean predicate and a universe-valued term
# function. Such a symbol cannot be given one coherent Z3 declaration, and the
# program crashes at exec time ("'ExprRef' object is not callable", "Wrong
# number of arguments", "Sort mismatch").
#
# Rather than discard the whole record, we detect the conflicting symbols and
# drop only the formulas that use them in a non-canonical way, then solve with
# the consistent remainder. When a symbol is ever used as a bare individual we
# prefer the individual reading (it is the most common intent, e.g. a named
# person) and drop formulas that apply it like a function.


def _walk_symbol_usages(
    node: Node, usages: dict[str, set[tuple[str, int]]], in_term: bool,
    locals_: set[str],
) -> None:
    """Record, per symbol, the set of (role, arity) pairs it is used with.

    role is 'pred' (Boolean-position application), 'func' (term-position
    application), or 'const' (bare individual). Bound variables are ignored.
    """
    if node is None or isinstance(node, (Var, Num)):
        return
    if isinstance(node, Const):
        if node.name not in locals_:
            usages.setdefault(node.name, set()).add(("const", 0))
        return
    if isinstance(node, App):
        role = "func" if in_term else "pred"
        usages.setdefault(node.name, set()).add((role, len(node.args)))
        for a in node.args:
            _walk_symbol_usages(a, usages, True, locals_)
        return
    if isinstance(node, Not):
        _walk_symbol_usages(node.body, usages, in_term, locals_)
        return
    if isinstance(node, BinOp):
        _walk_symbol_usages(node.left, usages, in_term, locals_)
        _walk_symbol_usages(node.right, usages, in_term, locals_)
        return
    if isinstance(node, (Cmp, Arith)):
        _walk_symbol_usages(node.left, usages, True, locals_)
        _walk_symbol_usages(node.right, usages, True, locals_)
        return
    if isinstance(node, Quant):
        _walk_symbol_usages(node.body, usages, in_term, locals_ | set(node.vars))
        return


def _resolve_canonical(
    all_usages: dict[str, set[tuple[str, int]]],
    goal_usages: dict[str, set[tuple[str, int]]],
) -> dict[str, tuple[str, int] | None]:
    """For every symbol used inconsistently across the record, choose the single
    (role, arity) reading to keep; formulas using it any other way are dropped.

    A symbol is conflicted when it is used with more than one arity, or mixes a
    bare-individual reading with a functional one, or mixes Boolean-predicate and
    universe-function readings.

    The canonical reading is chosen to keep the *goal* solvable:
      * if the goal uses the symbol with a single reading, that reading wins
        (the goal is the query — never sacrifice it to a stray premise);
      * else if it is ever used as a bare individual, keep the individual reading;
      * else there is no safe reading, and every formula using it is dropped.
    """
    canonical: dict[str, tuple[str, int] | None] = {}
    for name, role_set in all_usages.items():
        arities = {a for (_, a) in role_set}
        roles = {r for (r, _) in role_set}
        is_conflict = (
            len(arities) > 1
            or ("const" in roles and roles != {"const"})
            or ("pred" in roles and "func" in roles)
        )
        if not is_conflict:
            continue
        goal_reading = goal_usages.get(name)
        if goal_reading is not None and len(goal_reading) == 1:
            canonical[name] = next(iter(goal_reading))
        elif "const" in roles:
            canonical[name] = ("const", 0)
        else:
            canonical[name] = None
    return canonical


def _node_conflicts(
    node: Node, canonical: dict[str, tuple[str, int] | None]
) -> bool:
    """Whether `node` uses any conflicted symbol in a non-canonical way (and so
    must be dropped). A symbol with canonical reading None is unusable anywhere."""
    node_usages: dict[str, set[tuple[str, int]]] = {}
    _walk_symbol_usages(node, node_usages, False, set())
    for name, canon in canonical.items():
        uses = node_usages.get(name)
        if not uses:
            continue
        if canon is None or any(u != canon for u in uses):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Top-level: convert a list of FOL formulas into a full Z3 Python program
# ─────────────────────────────────────────────────────────────────────────


def convert_premises_to_z3py(
    fol_premises: Iterable[str],
    goal_fol: str | None = None,
) -> tuple[list[str], list[str], str | None, list[int]]:
    """Convert a batch of FOL formulas (and an optional goal) to a Z3 Python
    program shape.

    Returns (setup_lines, premise_exprs, goal_expr_or_None, skipped_indices).
    Each premise_expr is a Z3 Python expression string suitable for
    `premises = [<expr_0>, <expr_1>, ...]`.

    Records that fail to parse are skipped — their indices are returned.
    """
    # Parse every formula first (signature collection is deferred until after
    # conflict resolution, so inconsistent formulas don't pollute declarations).
    premise_nodes: list[Node | None] = []
    skipped: list[int] = []
    for i, p in enumerate(fol_premises):
        try:
            premise_nodes.append(parse(p))
        except FolParseError:
            skipped.append(i)
            premise_nodes.append(None)

    goal_node: Node | None = None
    if goal_fol is not None:
        try:
            goal_node = parse(goal_fol)
        except FolParseError:
            goal_node = None

    # Detect symbols used inconsistently across the record (predicate vs constant,
    # clashing arities, Bool-predicate vs universe-function) and drop only the
    # formulas that use them non-canonically, keeping the consistent remainder.
    all_usages: dict[str, set[tuple[str, int]]] = {}
    for n in premise_nodes:
        if n is not None:
            _walk_symbol_usages(n, all_usages, False, set())
    goal_usages: dict[str, set[tuple[str, int]]] = {}
    if goal_node is not None:
        _walk_symbol_usages(goal_node, goal_usages, False, set())
        for name, rs in goal_usages.items():
            all_usages.setdefault(name, set()).update(rs)
    canonical = _resolve_canonical(all_usages, goal_usages)
    if canonical:
        premise_nodes = [
            None if (n is not None and _node_conflicts(n, canonical)) else n
            for n in premise_nodes
        ]
        if goal_node is not None and _node_conflicts(goal_node, canonical):
            goal_node = None

    # Collect the signature from the surviving (consistent) formulas only.
    sig = Signature()
    for n in premise_nodes:
        if n is not None:
            collect_signature(n, sig)
    if goal_node is not None:
        collect_signature(goal_node, sig)

    # If any premise contains a free variable that wasn't bound, declare it as
    # a fresh constant so the program still type-checks. (Some dataset items
    # write `∀x P(x)` correctly but others write `P(x)` at top level.)
    extra_free: set[str] = set()
    for n in premise_nodes:
        if n is not None:
            extra_free |= collect_free_vars(n)
    if goal_node is not None:
        extra_free |= collect_free_vars(goal_node)
    for v in extra_free:
        sig.constants.add(v)

    # Infer which symbols are numeric (RealSort) before rendering declarations,
    # so arithmetic and numeric-argument premises type-check under Z3.
    inference_nodes = [n for n in premise_nodes if n is not None]
    if goal_node is not None:
        inference_nodes.append(goal_node)
    infer_numeric_sorts(inference_nodes, sig)

    setup = render_setup(sig)
    premise_exprs = [render_expr(n, sig) for n in premise_nodes if n is not None]
    goal_expr = render_expr(goal_node, sig) if goal_node is not None else None
    return setup, premise_exprs, goal_expr, skipped
