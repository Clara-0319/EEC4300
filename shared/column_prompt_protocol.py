"""Observation-independent vertical-column primitives; model emits full layers."""
from combined_prompt_protocol import METHODS, STRATIFIED_METHODS, LIBRARY_METHODS, build_prompt
from combined_prompt_protocol import make_library as previous_library

def column_blocks(max_height):
    if type(max_height) is not int or max_height < 1:
        raise ValueError('max_height must be a positive integer')
    return ['1' * height + '0' * (max_height-height) for height in range(max_height+1)]

def make_library(task, difficulty):
    if task != '3d':
        return previous_library(task, difficulty)
    blocks = column_blocks(3)
    prompt = (
        '\nFixed pre-generated library of vertical column blocks of maximum height 3:\n'
        '000: empty column, height 0, contributes 0 voxels\n'
        '100: bottom occupied only, height 1, contributes 1 voxel\n'
        '110: bottom and middle occupied, height 2, contributes 2 voxels\n'
        '111: bottom, middle and top occupied, height 3, contributes 3 voxels\n'
        'Each 3-bit string describes ONE VERTICAL COLUMN at a fixed (row, column), '
        'NOT a horizontal row. Bits are ordered BOTTOM TO TOP: bit 1 is Layer 1, '
        'bit 2 is Layer 2, bit 3 is Layer 3. Do not reverse this order.\n'
        'Construct ONE complete structure by choosing exactly one column block for each '
        'of the 9 positions on the 3x3 grid. Blocks may be reused at different positions. '
        'Where the observed top view is 0, choose 000. Where it is 1, choose 100, 110 or 111. '
        'Keep the same row and column coordinates when assembling layers. '
        'At position (r,c), Layer k receives the kth bit of the chosen vertical column. '
        'Use only the four listed column blocks.\n'
        'Complexity remains the TOTAL NUMBER OF OCCUPIED VOXELS, equal to the sum '
        'of the chosen column heights, NOT the number of nonempty columns or layers. '
        'If a target complexity was specified above, the sum of column heights must equal it exactly. '
        'Satisfy all observations and avoid previous identical full structures.\n'
        'Assemble the columns yourself and return ONLY ONE complete Structure in the required '
        'layer-matrix format: Structure:, Layer 1:, three rows of three space-separated 0/1 cells, '
        'then each additional occupied layer in bottom-to-top order. '
        'Omit trailing all-zero layers. Do not output column IDs, a column-code grid, '
        'a height map, intermediate selections or multiple candidate answers.'
    )
    return {'task':task,'difficulty':difficulty,'blocks':blocks,'count':len(blocks),
            'prompt':prompt,'uses_observations':False,'uses_ground_truth':False,
            'api_calls_to_generate':0,'prompt_version':'vertical_columns_v2',
            'max_height':3,'bit_order':'bottom_to_top','complexity':'total_occupied_voxels'}
