from dash import Dash, dcc, html, Output, Input, State, ClientsideFunction, ctx
import dash_bootstrap_components as dbc
import torch
from text_encoding import Embedding
from model import BigramLanguageModel

colors = {"background": "#1D2630", "text": "#7FDBFF"}

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

emb = Embedding()
model = BigramLanguageModel(emb.vocab_size, n_embd, block_size, n_head, n_layer, dropout, device)
model.load_state_dict(torch.load("model.ptm"))

app = Dash(
    __name__,
        meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
    external_stylesheets=[dbc.themes.DARKLY],
    )

app.layout = html.Div(
    children=[
        html.Div(children=[
            html.H1(children="Custom Transformer model"),
            dcc.Input(id="user_input", type="text", placeholder="Ask about anything...", debounce=True, style={'height': '200px', 'width': '100%'}),
        ],
        style={'display':'inline-block', 'width':'40%',}),
        html.Div(id='output', style={'display':'inline-block', 'width':'55%'})
    ]
)

@app.callback(
    Output('output', 'children'),
    Input('user_input', 'value')
)
def get_usertext(prompt):
    if prompt is None:
        return
    context = emb.encode(prompt)
    model.cuda()
    generated_output = model.generate(torch.tensor(context).reshape((1, len(context))), max_new_tokens=256)
    return emb.decode(generated_output[0].tolist())

if __name__ == "__main__":
    app.run(port=8050)