# sign-language-recognition
A project for university project in Worclaw in module "Advanced Topics of AI".

## Visualizing the model architecture
The model architecture is visualized using `nnviz`.
```
pip install nnviz
```

On MacOS I had to set the environment variable before building the graphics.
```
export PYTHONPATH=$(pwd)
```

Then visualization can be run like this:
```
nnviz src.models.simple_cnn:BaselineCNN --style show_specs=False --style show_node_arguments=False --style show_node_params=False --style show_node_source=False --out report_visualizations/baseline_architecture.png
```
All flags here are optional.
`--style/-S` flag allows for customization. 
`--out/-o` flag is optional and states the path to save the visualization. Standard output is a PDF.
For more detailed information, see https://nnviz.readthedocs.io/en/latest/cli/customization.html.

To make the plot horizontal I make a `.dot` file and then convert it to `.svg` through `Graphviz`.
```
nnviz src.models.simple_cnn:BaselineCNN -S show_specs=False -S show_node_name=False -S show_node_params=False -S show_node_arguments=False -S show_node_source=False -S show_clusters=False -o report_visualizations/baseline_architecture.dot
```
```
dot -Grankdir=LR -Tsvg report_visualizations/baseline_architecture.dot -o report_visualizations/baseline_architecture.svg
```

dot -Grankdir=LR -Tsvg report_visualizations/regularized_architecture.dot -o report_visualizations/regularized_architecture.svg