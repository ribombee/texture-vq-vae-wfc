import os
# Hoping the below line stops a very annoying issue with ctrl c bricking pycharm : )
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
import random
import math
import pickle
import numpy as np
import time
#from sty import fg, bg, ef, rs, Style, RgbFg

def generate_new_level(height, width, model, wrapping=False, max_attempts = 5, iteration_levels = 1):
	
	pattern_occurrences = model["pattern_counts"]
	possible_patterns = list(pattern_occurrences.keys())
	allowed_adjacencies = model["allowed_adjacencies"]

	domain = model["domain"]

	i=0
	center = None
	iteration_width_square_step = height ** 2 / iteration_levels
	iteration_height_square_step = height ** 2 / iteration_levels
	iteration_width = 0
	iteration_height = 0 # Initialing height and width to be added to in steps.
	while i < max_attempts:
		for iteration_idx in range(iteration_levels):
			if iteration_idx == iteration_levels - 1: # We are making the final generation, which must match the original width and height
				iteration_width = width
				iteration_height = height
			else:
				iteration_width = int(math.sqrt(iteration_width**2 + iteration_width_square_step))
				iteration_height = int(math.sqrt(iteration_height**2 + iteration_height_square_step))
			start_time = time.time()
			print(f"Generating {iteration_width}x{iteration_height} pattern")
			level = initialize_level(iteration_height, iteration_width, possible_patterns, center)
			level = propagate(level, possible_patterns, allowed_adjacencies,
							  pattern_occurrences, wrapping) # Level must now be immediately propagated due to the insertion of observed pattern.

			possible_positions = get_observable_positions(level, pattern_occurrences)

			while len(possible_positions) > 0:
				pos, pat = observe(level, pattern_occurrences, possible_positions)

				level[pos[0]][pos[1]] = [pat]


				level = propagate(level, possible_patterns, allowed_adjacencies,
													pattern_occurrences, wrapping)

				possible_positions = get_observable_positions(level, pattern_occurrences)

				print(".", end = " ") #Printing a line of dots as we progress to show the runtime

			end_time = time.time()
			time_elapsed = end_time - start_time
			print("") # Getting newline for the following print
			print(f"Generating {iteration_width}x{iteration_height} pattern took {time_elapsed} seconds")

			if not is_valid_level(level):
				print(f"Contradiction reached during sampling. A position in the level "+
					f"has 0 possible patterns. Generation attempt {i} failed.")
				i+=1
				break
			else:
				center = level

	return finalize_level(level)

def initialize_level(height, width, possible_patterns, center = None):

	level = [[possible_patterns for column in range(width)]
									for row in range(height)]

	if center is not None:
		print(f"Inserting center: {center}")
		#Insert a pre-generated level/pattern where there
		center_height = len(center)
		center_width = len(center[0])

		center_height_offset = (height - center_height) //2
		center_width_offset = (width - center_width) // 2
		for row in range(center_height):
			for column in range(center_width):
				level[center_width_offset + row][center_height_offset + column] = center[row][column]
				#level[center_height_offset: center_height_offset + center_height][center_width_offset: center_width_offset + center_width] = center

	return level

# check if there are any positions with 0 options available
def is_valid_level(level):
	for row in level:
		for cell in row:
			if len(cell) == 0:
				return False

	return True

def pattern_to_tuple(pattern):
	flattened_pattern = [tile for row in pattern for tile in row]
	pattern_as_tuple = tuple(flattened_pattern)

	return pattern_as_tuple

def get_observable_positions(level, pattern_occurrences):
	# gather the positions with the fewest available options
	lowest_entropy = float("inf")
	possible_positions = []
	for row_index in range(len(level)):
		for col_index in range(len(level[row_index])):
			
			#print(f"{row_index},{col_index} = {level[row_index][col_index]}") # FINE
			#print(pattern_occurrences) # FINE
			entropy = compute_shannon_entropy(level[row_index][col_index],
														pattern_occurrences)

			# either a fail state (no options), or a collapsed state (1 option)
			if entropy == 0:
				if len(level[row_index][col_index]) == 0:
					print("Ran into a fail case; no options available for a "+
						"position. Restarting generation.")
					return []
				else:
					# not a fail state, this position is just already collapsed
					continue
			# new lowest entropy position found, overwrite possible positions
			elif entropy < lowest_entropy:
				lowest_entropy = entropy
				possible_positions = [[row_index,col_index]]

			# position with the same as current lowest entropy,
			# append to possible positions
			elif entropy == lowest_entropy:
				possible_positions.append([row_index,col_index])
			
			# entropy is higher than current lowest entropy, skip position
			else:
				continue

	return possible_positions

def observe(level, pattern_occurrences, possible_positions):
	# randomly choose which position to collapse
	position = random.choice(possible_positions)
	
	# get the possible patterns at the chosen position
	possible_patterns_at_position = level[position[0]][position[1]]

	# construct a weighted choice for those patters based on occurrences
	weights = [pattern_occurrences[pattern] for pattern in possible_patterns_at_position]

	total_weight = sum(weights)
	weights=[weight/total_weight for weight in weights]

	chosen_pattern = random.choices(possible_patterns_at_position, 
									weights=weights, 
									k=1)[0]

	return position, chosen_pattern

def compute_shannon_entropy(patterns, occurrences):
	#pattern_counts = [occurrences[pattern_to_tuple(pat)] for pat in patterns]
	pattern_counts = [occurrences[pat] for pat in patterns]	
 
	total = sum(pattern_counts)
	pattern_counts = [count/total for count in pattern_counts]

	shannon_entropy = -sum([count*math.log(count) for count in pattern_counts])

	return shannon_entropy

# for every position, 
	#	get the patterns allowed at that position
	#	get the patterns allowed at surrounding positions given those patterns
	#	remove at patterns at the surrounding positions that are not allowed
	#	repeat this while any changes are made to the allowed patterns 
def propagate(level, patterns, allowed_adjacencies, pattern_occurrences, wrapping):
	still_updating = True
	i=0
	while still_updating:
		i+=1
		still_updating = False

		#get all positions sorted by entropy

		positions = [(r,c) for r in range(len(level)) 
										for c in range(len(level[0]))]

		# print("Computing entropy")
		sorted_positions = []
		for pos in positions:
			entropy = compute_shannon_entropy(level[pos[0]][pos[1]],
												pattern_occurrences)

			if len(sorted_positions) == 0:
				sorted_positions.append((pos,entropy))
			else:
				index = 0
				while index < len(sorted_positions) and \
										entropy > sorted_positions[index][1]:
					index += 1
				if index < len(sorted_positions):
					sorted_positions.insert(index, (pos,entropy))
				else:
					sorted_positions.append((pos,entropy))

		# print("Determining allowed adjs:")
		for pos,_ in sorted_positions:
			# print("\tiniting")
			r = pos[0]
			c = pos[1]

			allowed_above = []
			if wrapping:
				current_allowed_above = {pat for pat in level[(r-1)%len(level)][c]}
			elif not wrapping and r > 0:
				current_allowed_above = {pat for pat in level[r-1][c]}
			# if not wrapping, assume anything can be placed out of bounds
			elif not wrapping and r <= 0:
				current_allowed_above = {pat for pat in patterns}
			
			allowed_below = []
			if wrapping:
				current_allowed_below = {pat for pat in level[(r+1)%len(level)][c]}
			elif not wrapping and r < len(level)-1:
				current_allowed_below = {pat for pat in level[r+1][c]}
			# if not wrapping, assume anything can be placed out of bounds
			elif not wrapping and r >= len(level):
				current_allowed_below = {pat for pat in patterns}

			allowed_left = []
			if wrapping:
				current_allowed_left = {pat for pat in level[r][(c-1)%len(level[r])]}
			elif not wrapping and c > 0:
				current_allowed_left = {pat for pat in level[r][c-1]}
			# if not wrapping, assume anything can be placed out of bounds
			elif not wrapping and c <= 0:
				current_allowed_left = {pat for pat in patterns}

			allowed_right = []
			if wrapping:
				current_allowed_right = {pat for pat in level[r][(c+1)%len(level[r])]}
			elif not wrapping and c < len(level[0])-1:
				current_allowed_right = {pat for pat in level[r][c+1]}
			# if not wrapping, assume anything can be placed out of bounds
			elif not wrapping and c >= len(level[0]):
				current_allowed_right = {pat for pat in patterns}

			allowed_above = {pat_above for pat_curr in level[r][c] for pat_above in allowed_adjacencies[pat_curr]["above"]}
			allowed_below = {pat_below for pat_curr in level[r][c] for pat_below in allowed_adjacencies[pat_curr]["below"]}
			allowed_left = {pat_left for pat_curr in level[r][c] for pat_left in allowed_adjacencies[pat_curr]["left"]}
			allowed_right = {pat_right for pat_curr in level[r][c] for pat_right in allowed_adjacencies[pat_curr]["right"]}

			if wrapping:
				level[(r-1)%len(level)][c] = list(allowed_above.intersection(current_allowed_above))
				level[(r+1)%len(level)][c] = list(allowed_below.intersection(current_allowed_below))
				level[r][(c-1)%len(level[r])] = list(allowed_left.intersection(current_allowed_left))
				level[r][(c+1)%len(level[r])] = list(allowed_right.intersection(current_allowed_right))

				if len(level[(r-1)%len(level)][c]) < len(current_allowed_above) or \
					len(level[(r+1)%len(level)][c]) < len(current_allowed_below) or \
					len(level[r][(c-1)%len(level[r])]) < len(current_allowed_left) or \
					len(level[r][(c+1)%len(level[r])]) < len(current_allowed_right):
					still_updating = True
			else:
				if r > 0:
					level[r-1][c] = list(allowed_above.intersection(current_allowed_above))
					if len(level[r-1][c]) < len(current_allowed_above):
						still_updating = True
				if r < len(level)-1:
					level[r+1][c] = list(allowed_below.intersection(current_allowed_below))
					if len(level[r+1][c]) < len(current_allowed_below):
						still_updating = True
				if c > 0:
					level[r][c-1] = list(allowed_left.intersection(current_allowed_left))
					if len(level[r][c-1]) < len(current_allowed_left):
						still_updating = True
				if c < len(level[0])-1:
					level[r][c+1] = list(allowed_right.intersection(current_allowed_right))
					if len(level[r][c+1]) < len(current_allowed_right):
						still_updating = True
		
	return level



def finalize_level(level):

	final_level = [[cell[0][0] for cell in row] for row in level]

	return final_level


if __name__ == '__main__':
	# domain = "SMB"
	# domain = "LR"
	domain = "colors"
	textureLocation = "Textures/Texture0.npy" # trained model to load

	if domain == "SMB":
		wrapping = False
		level_height = 14
		level_width = 16

	elif domain == "LR":
		wrapping = True
		level_height = 16
		level_width = 16

	elif domain == "colors":
		wrapping = True
		level_height = 32
		level_width = 32

	trained_model = pickle.load(open(f"trained_WFC_"+textureLocation[9:-4]+".pickle", "rb"))


	level = generate_new_level(level_height, level_width, trained_model, 
											wrapping=wrapping, max_attempts=20)

	# print_level_in_progress(level, trained_model["domain"])
	numpyfile = np.array(level) # level generated to numpy array
	np.save('output/generated'+textureLocation[9:-4] , numpyfile)	#saving the numpy in the output

	print("holamundo")
	with open('output/generated.txt', 'w') as output:
		for row in level:
			for cell in row:
				output.write(f"{cell:^3},")
			output.write('\n')
