def which_sim(num):
    for K in range(len(training_list)):
        if num < train_num_particles[sims.index(training_list[K])]:
            return K, num
        else:
            num -= train_num_particles[sims.index(training_list[K])]

if __name__ == '__main__':
    sims = [1,2,4,5,6,7,8,9]
    training_list = [1,2,4,8]
    train_num_particles = [2,3,4,5,6,7,8,9]

    training_num = 0
    for training_sim in training_list:
        training_num += train_num_particles[sims.index(training_sim)]

    for i in range(training_num):
        print(which_sim(i))
