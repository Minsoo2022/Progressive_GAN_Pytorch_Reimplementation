def add_images(board, images, stage, iteration, name) :
    #cliping
    board.add_images(str(stage)+'/'+name, (images / 2 + 0.5).clamp(0,1), iteration)

def add_losses(board, losses, stage, iteration) :
    keys = losses.keys()
    for key in keys :
        board.add_scalar(str(stage)+'/'+key, losses[key].mean(), iteration)