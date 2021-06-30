def pathModel(model):
    getpath = {
        'tdnn': 'time delay neural network',
        'tdnn_pso': 'time delay neural network + particle swarm optimization',
        'rf': 'random forest',
        'svm': 'support vector machine'
        }
    return getpath[model]