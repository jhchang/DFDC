import click
import Video_prediction

@click.command()
@click.option("--modelname", default="Xception", help="""Choose an architecture between
- EfficientNetB4
- EfficientNetB4ST
- EfficientNetAutoAttB4
- EfficientNetAutoAttB4ST
- Xception"

The default is Xception
)
""")
# @click.option("--modelname", prompt="model name: ", help="""Choose an architecture between
# - EfficientNetB4
# - EfficientNetB4ST
# - EfficientNetAutoAttB4
# - EfficientNetAutoAttB4ST
# - Xception")
# """

def run_model(modelname):
    Video_prediction.run_nb(modelname)

if __name__ == '__main__':
    run_model()