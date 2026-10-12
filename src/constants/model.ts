// Which GPT-2 size this site runs. Change this (and MODEL_NAME in the
// GitHub Actions workflow) to switch models; everything else reads from here.
// The model files themselves are produced by scripts/build_model.py and
// live in static/model/<MODEL_NAME>/ (they are not committed to git).
export const MODEL_NAME: string = import.meta.env.VITE_MODEL_NAME || 'gpt2-medium';

// Longest prompt (in words) a student can type. GPT-2 itself accepts up to
// 1,024 tokens; this keeps the attention diagrams readable.
export const WORD_LIMIT = 40;
