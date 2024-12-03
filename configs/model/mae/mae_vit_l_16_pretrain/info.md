
INFO: This is 1021 version



Since the 1020 version is worse than 1015

- by comparing 1020 and 1019 (no jitter), the 1020 version is better, so maybe jitter is not a problem

- the wide scope is not needed? -> the 1021_narrow does not use the wide scope

<!-- - the SFHQ-T2I images are too fake? -> the 1021_no_sfhq version does not use the SFHQ-T2I images -->


## this: 1021 version
- ideally, the difference with 1015 version is just adding SFHQ-T2I, however, in 1015 verison, there was a bug where the VGGFace2 was not using jitter. 
- So, we cannot use 1021 version and 1015 version to compare the effect of SFHQ-T2I images.
- therefore, we run 1022 version (excluding SFHQ), see below.