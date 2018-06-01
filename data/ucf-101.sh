wget http://crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e UCF101.rar ucf101-videos/
rm UCF101.rar

for video in `ls ucf101-videos`; do
  mkdir -p ucf101-frames/$video
  ffmpeg -i ucf101-videos/$video -vf fps=1 ucf101-frames/$video/%04d.jpg
done
