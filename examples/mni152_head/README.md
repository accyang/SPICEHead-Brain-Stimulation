# Example head: CHARM-segmented MNI152 template

`labels_1mm.nii.gz` is the ten-tissue segmentation of the MNI152 (ICBM 2009) T1 template produced
by SimNIBS 4.6 CHARM at 0.5 mm and resampled here to 1 mm isotropic by nearest neighbour. Label
values follow CHARM (see `spicehead/conductivity.py`, `CHARM_LABELS`): 1 white matter, 2 grey
matter, 3 CSF, 4 bone, 5 scalp, 6 eyeballs, 7 compact bone, 8 spongy bone, 9 blood, 10 muscle;
0 is air.

`electrodes_1020_mni.json` gives the 19 positions of the 10-20 system in MNI space, taken from the
cap file `EEG10-10_UI_Jurak_2007.csv` distributed with SimNIBS (Jurcak V, Tsuzuki D, Dan I.
NeuroImage 2007;34:1600-11). `roi_coords_mni.json` gives the six target centres used in the paper
(mm, MNI); each target is a 10 mm sphere about its centre(s).

This is a template, not a subject. No participant data is included. The MNI152 template is
copyright McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University, and
is redistributed under its licence (free use and redistribution with attribution).

Run the paper's search on it:

    python examples/mni152_full_search.py --dx 3      (a few minutes on an idle workstation)
    python examples/mni152_full_search.py --dx 2      (tens of minutes)

ICBM 152 nonlinear atlases version 2009. Copyright (C) 1993–2004 Louis Collins,
McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University.
The [template licence and redistribution terms](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009#License)
apply to the underlying template.
