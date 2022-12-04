import pickle
import os

# dict1 = pickle.load(open("num_patterns_per_dim.pickle","rb"))
# dict2 = pickle.load(open("num_patterns_per_dim_v2.pickle","rb"))
# dict3 = pickle.load(open("num_patterns_per_dim_v3.pickle","rb"))


# print(dict1.keys())

# dict1_reformatted = {(128,16):[]}

# for t in dict1[(128,16)]:
#     dict1_reformatted[(128,16)].append(t[1])
    
# dict2.update(dict1_reformatted)
# dict2.update(dict3)

# pickle.dump(dict2, open("num_patterns_per_dim_combined.pickle", "wb"))

d = pickle.load(open("num_patterns_per_dim_combined.pickle","rb"))

#print(d[(128,16)])

avg = lambda l: sum(l)/len(l)

latent_dims = [16,32,64,128,256,512]
num_embeds = [8,16,32,64]

print_line = lambda: print("----" + "+----------" * len(latent_dims))

print()
print(f"    | {' | '.join(f'{n:>8d}' for n in latent_dims)}")
print_line()
for ne in num_embeds:
    val_strings = [f'{avg(d[(ld,ne)]):>8.3f}' for ld in latent_dims]
    print(f"{ne:>3d} | {' | '.join(val_strings)}")
    print_line()
print()

# for k in list(d.keys()):
#     print(f"Model = {k}, average = ", end='')
#     average = avg(d[k])
#     print(f"{average:.3f}")
