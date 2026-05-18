"""Small compatibility patch for KIF prompts.jsonl.

KIF prompt records may store the supervised answer under `expected` rather than
`response` or `answer`. ReGLU's flexible loader reads the tuple below at runtime,
so extending it here is enough for both the dev dataset check and the main run.
"""

from methods import reglu_patch

reglu_patch._RESPONSE_KEYS = (
    "response",
    "answer",
    "expected",
    "completion",
    "output",
    "target",
    "label",
)
