"""
The combine operation of the Push Implementation (refer Turing Beads.good-notes)

In this operation, the RGB stimulus is converted into a lightweight "beehive" like structure. This structure
is analogous to a set of coloured rain drops separated by a membrane.

After conversion, each unit / "drop" is combined with its neighbours based on similarity. The membrane
is dissolved, and they form a larger "drop".

In the fovea, make sure to use principles from loop subdivision to enrich the resolution of images as follows:
    1. Let's suppose there are three photons carrying different colours A, B and C
    2. If there exists only two cones to absorb those three photos, usually there's a merge between photon A
        & B, and B & C
    3. To extract the original tri-colours out, need to find B. However there is an assumption here that there
    exists enough of A & C to individually remove them from the boundary colour

Alternatively, use unsharp masking principles to increase brightness / "energy" at the boundaries. Horizontal
cells spread their signals laterally to inhibit nearby photoreceptors. This “lateral inhibition” improves
contrast and sharpens edges. Essentially, after bipolar cell processing, horizontal cells makes the stimulus
interact constructively / destructively so that "context" spreads out horizontally.

"""

'''
In a traditional ML sense, this operation is equivalent to channel preprocessing in a
Convolutional Neural Network (CNN).

Consider this datastructure:

{
    area: NUMBER, provides flexibility on the shape of the drop without describing the exact
        shape, which is of secondary significance
    *pointers: LIST of all neighbours, but carefully mapped to a "sub number" in the area.
        This esoteric structure should be equivalent to the Turing implementation of a "drop".
    stimulus: DATA, contains the RGB transformed structure for stimulus.

REQUIREMENT: The "quality" of this operation should match or exceed one of CNN.
'''


def convert():
    pass


'''
In a traditional ML sense, this operation is equivalent to the convolutional operator
in a CNN. However, the analogy is made purely for understanding purposes. 

REQUIREMENT: The "quality" of this operation should match or exceed one of CNN.
'''


def combine():
    pass
