this is a soft-impute algorithm which is an algorithm to solve low rank matrix completion.
https://web.stanford.edu/~hastie/Papers/mazumder10a.pdf

```
.
├── main.py
├── make_matrix.py
├── methods_algorithm.py
├── methods_cal_param.py
└── methods_matrix.py
```

main.py is a code to set params and data and start algorithm.
make_matrix.py makes matrixes as input data.
methods_algorithm.py is a code for soft impute algorithm
methods_cal_param.py calculate parameters such as error terminal condition and so on.
methods_matrix.py contains manipulation functions such as to make matrix low rank.
