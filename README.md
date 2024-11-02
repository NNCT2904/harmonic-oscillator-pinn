# Harmonic Oscilator demonstration

Implementation of under damped harmonic oscilator:
- https://beltoforion.de/en/harmonic_oscillator/
- https://en.wikipedia.org/wiki/Harmonic_oscillator

In short, we simulate this with PINN:

<img src="Damped_spring.gif" width="50">

Experiment samples will be presented as png files, combine them into a clip with:

``` bash 
# within the plot subfolders
cat $(find . -name '*.png' | sort -V) | ffmpeg -hwaccel cuda -framerate 60 -i - -c:v libx264 -pix_fmt yuv420p -s 4000x800 out.mp4

# if use nvidia cards
cat $(find . -name '*.png' | sort -V) | ffmpeg -hwaccel cuda  -hwaccel_output_format cuda -framerate 60 -i - -c:v h264_nvenc -pix_fmt yuv420p -s 4000x800 out.mp4 
```

