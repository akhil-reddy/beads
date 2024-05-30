"""
The combine operation of the Push Implementation (refer Turing Beads.good-notes)

In this operation, the RGB image is converted into a lightweight "beehive" like structure. This structure
is analogous to a set of coloured rain drops separated by a membrane.

After conversion, each unit / "drop" is combined with its neighbours based on similarity. The membrane
is dissolved, and they form a larger "drop".

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
    stimulus: DATA, contains the RGB transformed structure for stimulus
}

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
