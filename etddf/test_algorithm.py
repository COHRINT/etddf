import numpy as np

matches = [1,2,3,4]
times = np.linspace(0,5,50)

def find_indices(explicit, implicit):
    assert len(implicit) >= len(explicit)
    size_diff = len(implicit) - len(explicit)
    indices = [x + size_diff for x in range(len(explicit))]

    for i in range(len(explicit)):
        start_ind = 0 if i == 0 else indices[i-1]
        end_ind = indices[i]

        search_times = np.array(implicit[start_ind:end_ind])
        diffs = np.abs( search_times - explicit[i] )
        best_ind = np.argmin(diffs) + start_ind
        indices[i] = best_ind

    return indices

indices = find_indices(matches, times)
for i in range(len(indices)):
    print(matches[i])
    print(times[indices[i]])