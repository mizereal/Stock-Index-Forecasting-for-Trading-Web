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

def get_plot(labely, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8, x9, y9):
    plt.switch_backend('AGG')
    plt.figure(figsize=(12,5))
    # plt.title(stock, fontsize=24)
    plt.plot(x1,y1, color='k', label ='Actual')
    plt.plot(x2,y2, color='b', label ='Trend_tdnn')
    plt.plot(x3,y3, color='r', label ='Predict_tdnn')
    plt.plot(x4,y4, color='g', label ='Trend_tdnn_pso')
    plt.plot(x5,y5, color='m', label ='Predict_tdnn_pso')
    plt.plot(x6,y6, color='purple', label ='Trend_rf')
    plt.plot(x7,y7, color='orange', label ='Predict_rf')
    plt.plot(x8,y8, color='brown', label ='Trend_svm')
    plt.plot(x9,y9, color='pink', label ='Predict_svm')
    plt.xlabel('Day')
    plt.xticks(fontsize=6)
    plt.ylabel(labely)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    graph = get_graph()
    return graph