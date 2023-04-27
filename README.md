# counterspeech

Auto Counter-Speech Generation for Hate Speech

Colaboratory Usage Example: <a target="_blank" href="https://colab.research.google.com/drive/1NSPWzqA96EAAWA-BEALhcKZJlUPxfKt4?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> 

## Usage Instructions (Minimal)

Install the latest version of the package for use on Colaboratory/DataBricks
```bash
pip install https://github.com/0xrutvij/counterspeech/releases/latest/download/counterspeech.tar.gz
```

For Compute Cluster Usage
```bash
git clone https://github.com/0xrutvij/counterspeech.git
cd counterspeech
pip install -e .
python main.py --run examples # or `./run.sh`
```

## Development Instructions (Minimal)

```bash
pip install -e .

pre-commit install
```
