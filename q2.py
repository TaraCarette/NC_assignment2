initVel = 10
initPos = 20

termLoops = 50

# first settings
w = 0.5
alpha1 = 1.5
alpha2 = 1.5
r1 = 0.5
r2 = 0.5

# second settings
# w = 0.7
# alpha1 = 1.5
# alpha2 = 1.5
# r1 = 1
# r2 = 1


def fitness(x):
    return x ** 2

def main():
    print("running")

    velocity = initVel
    position = initPos
    lBest = position
    gBest = position

    counter = 0
    while counter < termLoops:
        counter += 1
        print(position)

        # for each particle, which in this case is just one
        # do not need to create random vectors here

        # update velocity
        velocity = (w * velocity) + (alpha1 * r1 * (lBest - position)) + (alpha2 * r2 * (gBest - position))

        # update positions
        position = position + velocity

        # update local best
        if fitness(position) < fitness(lBest):
            lBest = position

        # update global best
        # as there is only a single particle the best is simply itself
        if fitness(position) < fitness(gBest):
            gBest = position



if __name__ == '__main__':
    main()