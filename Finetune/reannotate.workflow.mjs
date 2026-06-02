export const meta = {
  name: 'reannotate-nl-fol',
  description: 'Re-translate dataset NL premises/questions into faithful well-formed FOL, then review',
  phases: [
    { title: 'Translate', detail: 'NL -> FOL per batch of records' },
    { title: 'Review', detail: 'adversarial faithfulness/well-formedness check' },
  ],
}

const A = typeof args === 'string' ? JSON.parse(args) : (args || {})
const DIR = A.dir
const N = A.nBatches
if (!N || !DIR) {
  log(`BAD ARGS: typeof=${typeof args} DIR=${DIR} N=${N} raw=${JSON.stringify(args).slice(0, 200)}`)
}

const FOL_RULES = `FOL OUTPUT FORMAT (must parse in our Z3 converter):
- Quantifiers: write ∀x (...) and ∃x (...). ALWAYS parenthesize the body: ∀x (P(x) → Q(x)). For several variables nest or chain: ∀x (∃y (R(x, y))) or ∀x ∀y (...).
- Connectives: → (implies), ∧ (and), ∨ (or), ¬ (not), ↔ (iff).
- Predicates: CapWords or snake_case with arguments, e.g. WellTested(x), follows_pep8(x); nullary like Raining are allowed.
- Constants / named individuals: Capitalized (John, Math, IT). Variables: lowercase (x, y, s, m).
- Numbers & comparisons through value-functions: gpa(s) ≥ 2.0, grade(s, m) > 8.5, a = b, a ≠ b, plus ≤ < >.
- FORBIDDEN: modal operators (Possibly, Necessarily, SometimesButNotAlways), set/meta notation ({...}, Supports(...)), and free-text English. Express the underlying first-order claim instead.
- NO free variables: every variable must be bound by a quantifier. Use the SAME predicate names across a record's premises AND its questions so the solver can relate them.

RULES:
1. Produce exactly ONE FOL per NL premise, in the SAME ORDER (1:1). The length of premises_fol MUST equal the number of premises_nl.
2. The *_current fields are references: KEEP a formula unchanged when it faithfully and well-formedly captures the English; FIX it when it is wrong, malformed, misaligned, or non-first-order.
3. Each question_nl is usually "Based on the above premises, which statement can be inferred: CLAIM." Translate exactly CLAIM (the text after the colon) into one FOL using the record's predicate vocabulary.
4. Faithfulness first: preserve implication direction, every negation, and quantifier strength (all vs some) exactly as the English states.`

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    records: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          record: { type: 'integer' },
          premises_fol: { type: 'array', items: { type: 'string' } },
          questions: {
            type: 'array',
            items: {
              type: 'object',
              additionalProperties: false,
              properties: {
                row: { type: 'integer' },
                question_fol: { type: 'string' },
              },
              required: ['row', 'question_fol'],
            },
          },
        },
        required: ['record', 'premises_fol', 'questions'],
      },
    },
  },
  required: ['records'],
}

function fp(i) {
  return DIR + '/in_' + String(i).padStart(3, '0') + '.json'
}

function txPrompt(i) {
  return `You translate an educational-logic dataset from natural language into first-order logic.

Use the Read tool to read this UTF-8 JSON file of records:
${fp(i)}

It is a JSON array of records; each record has:
- premises_nl: English premises
- premises_fol_current: the current (possibly wrong) FOL, one per premise
- questions: list of { row, question_nl, question_fol_current }

For EVERY record and EVERY question in the file, produce corrected FOL.

${FOL_RULES}

Return { records: [ { record, premises_fol: [...one per NL premise...], questions: [ { row, question_fol } ] } ] } covering every record and every question in the file. The record and row integers MUST match the input exactly. Do not skip any.`
}

function rvPrompt(i, tx) {
  return `You are an adversarial reviewer of first-order-logic translations. Find and FIX faithfulness and well-formedness errors.

Use the Read tool to read the original English at:
${fp(i)}

Here is the proposed FOL to verify and correct:
${JSON.stringify(tx)}

For each record and question, compare the FOL against the English and correct any of: reversed implication direction; wrong quantifier (all vs some); missing or extra negation; predicate names that differ between the premises and the question (they must match); free variables; leftover modal/set/prose notation; or a premises_fol length that does not equal the number of premises_nl.

${FOL_RULES}

Return the same schema { records: [...] } with verified/corrected FOL for every record and question. Keep already-correct formulas unchanged. record and row integers MUST match the input.`
}

const idxs = Array.from({ length: N }, (_, i) => i)
log(`Re-annotating ${N} batches (translate -> review)`)

const results = await pipeline(
  idxs,
  (i) => agent(txPrompt(i), { label: `tx:${String(i).padStart(3, '0')}`, phase: 'Translate', schema: SCHEMA }),
  (tx, i) => agent(rvPrompt(i, tx), { label: `rv:${String(i).padStart(3, '0')}`, phase: 'Review', schema: SCHEMA }),
)

const batches = results.map((r, i) => (r && r.records ? { batch: i, records: r.records } : { batch: i, failed: true }))
const okCount = batches.filter((b) => !b.failed).length
log(`Completed ${okCount}/${N} batches`)
return { n_batches: N, ok: okCount, batches }
