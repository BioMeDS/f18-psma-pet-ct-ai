import nibabel as nib
import sys

for file in sys.argv[1:]:
    n = nib.load(file).get_fdata()
    print(file, n.min(), n.max(), sep="\t")