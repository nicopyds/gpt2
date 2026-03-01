from urllib import request

url_ = "https://gist.githubusercontent.com/jsdario/6d6c69398cb0c73111e49f1218960f79/raw/8d4fc4548d437e2a7203a5aeeace5477f598827d/el_quijote.txt"
request.urlretrieve(url_, "quijote.txt")

with open("quijote.txt") as f:
    quijote = f.read().split("\n")


palabras_interes = ["quijote"]
contador = 0

for line in quijote:
    for palabra in line.split():
        if palabra.lower() in palabras_interes:
            contador += 1

print(contador)


