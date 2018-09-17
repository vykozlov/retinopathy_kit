import requests

def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
         "predictions": [],
    }

    for label, prob in zip(labels, probabilities):
        name = label

        pred = {
            "label": name,
            "probability": float(prob),
            "info": {
                "links": [{"link": 'Google images', "url": image_link('diabetic retinopathy')},
                          {"link": 'Wikipedia', "url": wikipedia_link('retinopathy')}],
            },
        }
        d["predictions"].append(pred)
    return d

def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def format_train(network, accuracy, nepochs, data_size):
    d = {
        "status": "ok",
         "training": [],
    }
    
    train_info = {
        "network": network,            
        "test accuracy": accuracy,
        "n epochs": nepochs,
        "train set (images)": data_size['train'],
        "validation set (images)": data_size['valid'],
        "test set (images)": data_size['test'],
    }
    
    d["training"].append(train_info)
    return d
