# MSS

This library is responsable for recording the user's screen. It does by screenshoting frame by frame.

A simplles example of how to screenshot your first monitor at [TEST_mss.py](/Tutorials/TEST_mss.py)
```python
from mss import MSS

with MSS() as sct:
    sct.shot()
```

**Output:**
![Screencshot](monitor-1.png)