# Tracker: Using Kalman Filtering to track objects (Information 2020 Spring)

### (1) Setup

This code has been tested with Python 3.6.3, numpy 1.18.3, python-opencv 4.2.0, scipy 1.4.1 .

It also need some useful libraries: filterpy, imageio(opotion).

- install darknet(the newest yolov4 version)
It has been test with CUDA10.1 and cudnn7.6. Detailed install steps are all in `readme.md`. 
```ruby
# 1. modify as your need CMakeLists.txt or Makefile
$ ./build.sh
# or
$ make
```

- test the tracker demo
```ruby
$ python darknet_video.py
```

### (2) Demo

#### Persons' Track
---------
 <div align=center>
<img width="640" height="480" src="./data/person.gif"/>
</div>

#### Aeroplanes' Track
---------
 <div align=center>
<img width="640" height="480" src="./data/aeroplane.gif"/>
</div>

#### Boats' Track
---------
 <div align=center>
<img width="640" height="480" src="./data/boat.gif"/>
</div>


### Citation
If you find our work useful in your research, please consider citing:

	@project{information2020spring,
	  title={Tracker: Using Kalman Filtering to track objects},
	  author={Jin Yujie},
	  year={2020}
	}
