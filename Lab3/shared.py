from numpy import array

datasets = {
    'tram': {
        'img1': "data/green.png",
        'img2': "data/poster.png",
        'points1': [array([95, 170, 1]),
                    array([238,171, 1]),
                    array([235, 233, 1]),
                    array([94, 239,1]) ],
        'points2': [array([0, 0, 1]),
                    array([143, 0, 1]),
                    array([143, 66, 1]),
                    array([0, 66, 1])]
    },
    'stata': {
        'img1': "data/pano/stata-1.png",
        'img2': "data/pano/stata-2.png",
        
        'points1': [array([218, 209, 1]),
                    array([300,425, 1]),
                    array([337, 209, 1]),
                    array([336, 396,1])],
        'points2': [array([4, 232, 1]),
                    array([62, 465, 1]),
                    array([125, 247, 1]),
                    array([102, 433, 1])]
        
        }
}

def get_dataset(name):
    if name in datasets:
        return datasets[name]
    else:
        print(list(datasets.keys()))
        raise NameError("No dataset %s" % name)
