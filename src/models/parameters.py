import numpy as np
#from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

parameters_bagg = [
    {
        'base_estimator__clf__estimator': [LogisticRegression(random_state=11, max_iter=1000)],
        'base_estimator__clf__estimator__penalty': ['l1', 'l2'],
        'base_estimator__clf__estimator__C': np.logspace(-4, 4, 10),
        'base_estimator__clf__estimator__solver': ['liblinear']


    },
    {
        'base_estimator__clf__estimator': [KNeighborsClassifier()],
        'base_estimator__clf__estimator__n_neighbors': [4, 8, 15, 30],
        'base_estimator__clf__estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    },
    #{
    #    'base_estimator__clf__estimator': [SVC(probability=True)],
    #    'base_estimator__clf__estimator__C': [0.5, 0.7, 0.9],
    #    'base_estimator__clf__estimator__kernel': ['linear']
    #},
    {
        'base_estimator__clf__estimator': [DecisionTreeClassifier()],
        "base_estimator__clf__estimator__criterion": ["gini", "entropy"],
        "base_estimator__clf__estimator__max_depth": [4, 8, 12], 
        "base_estimator__clf__estimator__min_samples_leaf": [5, 25, 50, 100]
    },
    {
        'base_estimator__clf__estimator': [RandomForestClassifier()],
        "base_estimator__clf__estimator__n_estimators": [10, 20, 50, 100],
        "base_estimator__clf__estimator__min_samples_leaf": [10, 25, 50, 100],               
        "base_estimator__clf__estimator__max_depth": [2, 4, 5],
        "base_estimator__clf__estimator__min_samples_split": [10, 20, 50]
    },
    {
        'base_estimator__clf__estimator': [GradientBoostingClassifier()],
        "base_estimator__clf__estimator__n_estimators": [10, 20, 50, 100, 200],
        "base_estimator__clf__estimator__learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2, 0.5],
        "base_estimator__clf__estimator__max_depth": [3, 5, 8],
        "base_estimator__clf__estimator__subsample": [0.5, .75, 1.0]

    }
]

parameters_smote = [
    {
        'clf__estimator': [LogisticRegression(random_state=11, max_iter=1000)],
        'clf__estimator__penalty': ['l1', 'l2'],
        'clf__estimator__C': np.logspace(-4, 4, 10),
        'clf__estimator__solver': ['liblinear']
    },
    {
        'clf__estimator': [KNeighborsClassifier()],
        'clf__estimator__n_neighbors': [4, 8, 15, 30],
        'clf__estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']                                        
    },
    #{
    #    'clf__estimator': [SVC(probability=True)],
    #    'clf__estimator__C': [0.5, 0.7, 0.9],
    #    'clf__estimator__kernel': ['linear']
    #},
    {
        'clf__estimator': [DecisionTreeClassifier()],
        "clf__estimator__criterion": ["gini", "entropy"],
        "clf__estimator__max_depth": [4, 8, 12], 
        "clf__estimator__min_samples_leaf": [5, 25, 50, 100]
    },
    {
        'clf__estimator': [RandomForestClassifier()],
        "clf__estimator__n_estimators": [10, 20, 50, 100],
        "clf__estimator__min_samples_leaf": [10, 25, 50, 100],               
        "clf__estimator__max_depth": [2, 4, 5],
        "clf__estimator__min_samples_split": [10, 20, 50]
    },
    {
        'clf__estimator': [GradientBoostingClassifier()],
        "clf__estimator__n_estimators": [10, 50, 100, 200],
        "clf__estimator__learning_rate": [0.01, 0.1, 0.2, 0.5],
        "clf__estimator__max_depth": [3, 5, 8],
        "clf__estimator__subsample": [.75, 1.0]
    }
]



safra_best_parameters_bbagg = {
    "cenario_1": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=2, n_estimators=20)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ],

    "cenario_2": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=10, n_estimators=20)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)}
    ],

    "cenario_3": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.01)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=10, n_estimators=50)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ],

    "cenario_4": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.01)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=10, n_estimators=50)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ]
}

#   Safra Smote
safra_best_parameters_smote = {
    "cenario_1": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)}   ,
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=20, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}
    ],

    "cenario_2": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='brute', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=20)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}    
    ],

    "cenario_3": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=50)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ],

    "cenario_4": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=8, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=2, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}
    ]
}

safra_best_parameters_tomek = {
    "cenario_1": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ],

    "cenario_2": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='brute', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=2, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=5)}    
    ],

    "cenario_3": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='brute', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=2, n_estimators=20)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)}
    ],

    "cenario_4": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.01)},
        {'clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=15)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ]
}

# Shuffle

shuffle_best_parameters_bbagg = {
    "cenario_1": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=20, n_estimators=50)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.1, n_estimators=20, max_depth=3)}
    ],
    
    "cenario_2": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=50)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}
    ],

    "cenario_3": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.01)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=20, n_estimators=20)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}
    ],

    "cenario_4": [
        {'base_estimator__clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'base_estimator__clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'base_estimator__clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'base_estimator__clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=15)},
        {'base_estimator__clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=2, n_estimators=100)},
        {'base_estimator__clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}    
    ]

}


shuffle_best_parameters_smote = {
    "cenario_1": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.7, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=20, n_estimators=100)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}
    ],

    "cenario_2": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=10, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}    
    ],

    "cenario_3": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.01)},
        {'clf__estimator': KNeighborsClassifier(algorithm='brute', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=2, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)}
    ],

    "cenario_4": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='brute', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=50, min_samples_split=2, n_estimators=20)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=100, max_depth=3)}    
    ]
}

shuffle_best_parameters_tomek = {
    "cenario_1": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.9, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=10, n_estimators=20)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ],

    "cenario_2": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='auto', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=10, n_estimators=50)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=5)}
    ],

    "cenario_3": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=10, min_samples_split=10, n_estimators=10)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=60, max_depth=3)}    
    ],

    "cenario_4": [
        {'clf__estimator': LogisticRegression(random_state=11, max_iter=1000, C=0.1)},
        {'clf__estimator': KNeighborsClassifier(algorithm='ball_tree', n_neighbors=15)},
        #{'clf__estimator': SVC(kernel='linear', C=0.5, probability=True)},
        {'clf__estimator': DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=5)},
        {'clf__estimator': RandomForestClassifier(max_depth=2, min_samples_leaf=2, min_samples_split=2, n_estimators=50)},
        {'clf__estimator': GradientBoostingClassifier(learning_rate=0.01, n_estimators=20, max_depth=3)}
    ]
}
