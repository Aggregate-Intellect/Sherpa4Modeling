# Download the scenes for the CLEVR dataset.
curl https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip --output clevr.zip
unzip clevr.zip
mv CLEVR_v1.0/scenes/CLEVR_val_scenes.json data/CLEVR_val_scenes.json
rm -rf CLEVR_v1.0
rm clevr.zip

# Download the questions for the CLEVR dataset.
curl https://cs.stanford.edu/people/jcjohns/iep/CLEVR-Humans.zip --output clevr_humans.zip
unzip clevr_humans.zip
mv CLEVR-Humans/CLEVR-Humans-val.json data/CLEVR-Humans-val.json
rm -rf CLEVR-Humans
rm clevr_humans.zip