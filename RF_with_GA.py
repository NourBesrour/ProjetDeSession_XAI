import os
import cv2
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, algorithms
from tqdm import tqdm
import pickle
import h5py

path = 'brisc2025_balanced_aug/train'
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
imgDim = (50, 50, 3)

images = []
labels = []

for classIndex, className in enumerate(class_names):
    imageList = os.listdir(os.path.join(path, className))
    for imageFile in imageList:
        curImg = cv2.imread(os.path.join(path, className, imageFile))
        curImg = cv2.resize(curImg, (imgDim[0], imgDim[1]))
        gray = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = gray / 255.0
        images.append(gray.flatten())  # Flatten for RF
        labels.append(classIndex)

X = np.array(images)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, shuffle=True, stratify=y_train)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("n_estimators", random.choice, [50, 100, 200])
toolbox.register("max_depth", random.choice, [5, 10, 20, None])
toolbox.register("min_samples_split", random.randint, 2, 10)
toolbox.register("max_features", random.choice, ["sqrt", "log2", None])
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.n_estimators, toolbox.max_depth, toolbox.min_samples_split, toolbox.max_features), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_rf(individual):
    n_estimators, max_depth, min_samples_split, max_features = individual
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        max_features=max_features,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return (acc,)

toolbox.register("evaluate", evaluate_rf)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[50,5,2,0], up=[200,20,10,2], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=6)
NGEN = 2

for gen in range(NGEN):
    print(f"\n=== Generation {gen+1} ===")
    offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
    
    fits = []
    for ind in tqdm(offspring, desc="Evaluating individuals", ncols=100):
        fit = toolbox.evaluate(ind)
        fits.append(fit)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    best_in_gen = max(offspring, key=lambda ind: ind.fitness.values[0])
    print(f"Best validation accuracy this generation: {best_in_gen.fitness.values[0]:.4f}")

best_ind = tools.selBest(population, 1)[0]
print("\nBest hyperparameters found:", best_ind)

# -----------------------------
# Train final RF model
# -----------------------------
n_estimators, max_depth, min_samples_split, max_features = best_ind
rf_best = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    max_features=max_features,
    n_jobs=-1
)
rf_best.fit(X_train, y_train)

test_preds = rf_best.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
print(f"Test accuracy of best RF: {test_acc:.4f}")

import pickle

with open("RF_model.pkl", "wb") as f:
    pickle.dump(rf_best, f)

print("Random Forest model saved as RF_model.pkl")
