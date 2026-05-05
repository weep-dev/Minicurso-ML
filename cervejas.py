# %%
import pandas as pd

df = pd.read_excel("data/dados_cerveja.xlsx")
# %%
df.head()
# %%
features = ['temperatura', 'copo', 'espuma', 'cor']
target = 'classe'

X = df[features]
y = df[target]

# %%

from sklearn import tree

model = tree.DecisionTreeClassifier()
model.fit(X, y)

# %%
X = X.replace({
    "caneca":2,
    "americano":3,
    "não":0,
    "sim":1,
    "escura":4,
    "clara":5
})

# %%
model.predict([[-5, 2, 0, 4]])
# %%
import matplotlib.pyplot as plt

plt.figure(dpi=400, figsize=[4,4])

tree.plot_tree(model, feature_names=features, class_names=model.classes_, filled=True )
# %%
