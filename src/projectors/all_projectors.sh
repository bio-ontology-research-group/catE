set -e

python owl2vecstar_projector.py -i $1
python rdf_projector.py -i $1
python onto2graph_projector.py -i $1 -j ../projectors/
