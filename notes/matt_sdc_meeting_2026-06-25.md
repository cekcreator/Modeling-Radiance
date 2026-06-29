Matt needs:

1. Excel email next week
2. G/P Star - Figuring out how we get bins into libera utils
3. Test building the integration branch into a docker image and creating a manifest output.

Most useful to matt isMin: data cube from 3.7Would like: Data cube from m6 in loeb binsGold star: data cube from erbe scenes w/ m6Plat star: trmm bins + m6 data (edited)

Caveats to docker image are:excel/yaml is approved by others from matt/downstream users

cloud stuff is the same across the file, but different on a per file basis

open file, do processing, open out file, put in out file, close both files, open next file

First is get started on docker imagesecond is get m6 data cube workingThird ATBDFourth ml code with m6 data

in coef netcdf4 file add:

1. longer description of indexes for scene

2. explicitly list out the bins that are inclusive and the ranges 

in test_full_pipeline.py the data dirs will change as the SDC will add them to the SAE

want UNF-RAD-CAM In yaml file:

will get more variables like sza, vza, and raz passed through in algorithm.py

update to FMATCH instead of CAM