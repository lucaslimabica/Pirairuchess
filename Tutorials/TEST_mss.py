from mss import MSS

# The simplest use, save a screenshot of the 1st monitor
with MSS() as sct:
    sct.shot()