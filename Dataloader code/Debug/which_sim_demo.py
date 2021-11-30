
def which_sim(num):
    for K in range(len(training_list)):
        if num < train_num_particles[K]:
            return K, num
        else:
            num -= train_num_particles[K]

if __name__ == '__main__':
    sims = [1,2,3,4,5,6,7,8]
    training_list = [1,2,4,5]
    train_num_particles = [3,1,2,5]
    for i in range(sum(train_num_particles)):
        print(which_sim(i))
