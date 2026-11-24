"""Fixed-library and stratified-library prompts; no network or search."""
from matrix_adapters import make_library as original_library

STRATIFIED_METHODS = frozenset({'paper_stratified', 'stratified_library'})
LIBRARY_METHODS = frozenset({'prompt_library', 'stratified_library'})
METHODS = ['history_baseline', 'paper_stratified', 'prompt_library', 'stratified_library']

SUPPORT_RULE = (
    '\nMandatory support rule: Layer 1 is the bottom layer; layer numbers increase upward. '
    'For EVERY adjacent pair of layers and EVERY row r and column c, '
    'if L[k+1][r][c] = 1, then L[k][r][c] MUST equal 1. '
    'Equivalently, L[k+1][r][c] <= L[k][r][c]; for masks, (upper & lower) == upper. '
    'Support must be at the SAME row and column in the IMMEDIATELY preceding lower layer; '
    'a neighboring voxel or a voxel below a gap does not provide support. '
    'A lower 1 may have an upper 0, but a lower 0 must have an upper 0. '
    'Check every adjacent layer pair before returning ONE complete Structure.'
)

def make_library(task, difficulty):
    library = original_library(task, difficulty)
    if task == '3d':
        library['prompt'] += SUPPORT_RULE
    library['prompt_version'] = 'explicit_adjacent_support_v2' if task == '3d' else 'original_v1'
    return library

def build_prompt(adapter, history, method, library, target_complexity=None):
    if method not in METHODS:
        raise ValueError('Unknown method: ' + method)
    if (method in STRATIFIED_METHODS) != (target_complexity is not None):
        raise ValueError('Only stratified methods require a complexity target')
    prompt = adapter.prompt(history)
    if method in STRATIFIED_METHODS:
        prompt += f'\nThis proposal must have EXACTLY {target_complexity} units of the defined complexity. Match all observations and avoid previous equivalent answers.'
    if method in LIBRARY_METHODS:
        prompt += '\n' + library['prompt']
    return prompt
