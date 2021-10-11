import joblib

def Select(value):

    nm_img_config = joblib.load('imgConfs/nm_config.pkl')
    mlra_img_config = joblib.load('imgConfs/ml_config.pkl')
    brainT = joblib.load('imgConfs/brainT.pkl')

    optns = {"Pneumonia": nm_img_config,
             "Malaria": mlra_img_config,
             "Brain Tumor": brainT}

    return optns[value]
