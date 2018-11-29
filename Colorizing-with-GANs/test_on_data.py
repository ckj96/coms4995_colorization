from src import ModelOptions, main

options = ModelOptions().parse()
options.mode = 3
options.dataset = 'self'
options.checkpoint = './checkpoints/places365'
main(options)
