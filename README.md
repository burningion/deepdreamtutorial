# Deep Dream Tutorial

This is a walkthrough of the Deep Dream code created by Google. In it you'll learn how to create Deep Dreams, 
Controlled Deep Dreams, and Controlled Video Deem Dreams. There is a [blog post](http://www.kpkaiser.com/machine-learning/diving-deeper-into-deep-dreams/)
to go along with this, and pull requests are welcomed.

The images within were all created or shot by me, and you are free to do with them as you wish.

To get this running, first install caffe, ffmpeg, and opencv. That's a tall order, but luckily, Googling and hitting your head on a wall should help. Once you've done that:
```bash
$ ipython notebook
```

## Run the Video Example

```bash
$ cd video
$ ffmpeg -i les_coleman.m4v output%05d.jpg
$ python runnitoptical.py
$ cd output
$ ffmpeg -i output%05d.jpg out.mp4
```


Enjoy.
