import matplotlib.pyplot as plt
import base64
from io import BytesIO

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph

def get_plot(labely, x, y, xx, yy):
    plt.switch_backend('AGG')
    plt.figure(figsize=(12,5))
    # plt.title(stock, fontsize=24)
    plt.plot(x,y, label ='Actual')
    plt.plot(xx,yy, color='red', label ='Predict')
    plt.xlabel('Day')
    plt.xticks(fontsize=6)
    plt.ylabel(labely)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    graph = get_graph()
    return graph