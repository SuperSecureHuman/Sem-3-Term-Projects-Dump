# Given a input folder, read all the frames and make gif

import os
import imageio

def make_gif(input_path, output_path, duration=0.1):
    frames = []
    for i in range(1,1001):
        filename = os.path.join(input_path, f"{i}.jpg")
        frames.append(imageio.imread(filename))
    kargs = { 'duration': duration,
              'loop': 0 ,
              'subrectangles': True}

    imageio.mimwrite(output_path, frames, 'GIF', duration=duration,
                     loop=0, palettesize=16,  subrectangles=True)

make_gif('plots', 'plots/decision_boundry.gif')

