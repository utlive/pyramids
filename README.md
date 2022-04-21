This is a single file that allows you to use some functionality of the pyrtools library (https://pyrtools.readthedocs.io/) without having to import C libraries. The C libraries' importation can cause the failure of pyrtools in certain systems such as the Texas Supercomputers. The major changes that have been made are that corrDn and upConv have now been written as  pure python functions.

You can import steerable pyramids as :
```
from pyramids import SteerablePyramidSpace as SPyr
```

Most of the other functionality should work as well, although not all modules have been tested.
