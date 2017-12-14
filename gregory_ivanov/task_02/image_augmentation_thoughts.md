# Agreements

- Images are passed in the form "channels_last"

# Libraries for image augmentation

- [imgaug](https://github.com/aleju/imgaug)

# Actions in dataset:

## Notation

Code for an action has the following form: "ABC action_name (comment)" where
- A--: whether this actions is implemented in dataset or not. "D" means "implemented", "N" means "not implemented"
- -B-: is this action implemented well (IMHO). "W" means "implemented well", "B" means "implemented badly", "N" means "Don't know"
- --C: whether this action is useful in this particular task. "F" means "useFul", "L" means "useLess", "T" means "to be tested"


- DNT crop (unusual parameters like CROP_00 etc.)
- D
- N-- pad (imgaug)
- DBT fliplr (implemented for "channels_first" format)
- DBT flipud (implemented for "channels_first" format)
- N-- invert (imgaug)
- N-- add (imgaug)
- N-- multiply (imgaug)
- N-- gaussian_blur (imgaug)
- N-- average_blur (imgaug)
- N-- median_blur (imgaug)
- N-- bilateral_blur (imgaug)
- N-- gaussian_noise (imgaug)
- N-- Salt (imgaug)
- N-- Pepper (imgaug)
- N-- dropout (imgaug)
-

## additional realizations

- aplly an action to the particular area of an image (so called "coarsed methods")
