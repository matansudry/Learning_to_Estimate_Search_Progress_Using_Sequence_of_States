import pandas as pd
import tqdm
import os


def estimate_nodes(equation, max_d):
    cnt = 0
    total_sum = 0
    while 1:
        cnt += 1
        sum = (cnt*cnt*equation[0]+ cnt*equation[1] + equation[2])
        if ((equation[0] >= 0 and cnt > max_d) or cnt > (2*max_d)):
            break
        if (sum < 0):
            continue
        total_sum += sum
    return (total_sum)

def main():
    print(os.getcwd())
    main_path = "/data/preprocessing/20/gbfs_hff"
    ready_data_path = "/data/ready_data/40/gbfs_hff"
    domains =[
        ("airport", 30),
        ("blocks", 35),
        ("depot", 22),
        ("elevators", 30),
        ("freecell", 20),
        ("gripper", 20),
        ("logistics", 28),
        ("miconic", 30),
        ("movie", 30),
        ("openstacks", 30),
        ("parcprinter", 30),
        ("pegsol", 30),
        ("psr-small", 50),
        ("rovers", 30),
        ("satellite", 20),
        ("scanalyzer", 30),
        ("sokoban", 30),
        ("tpp", 30),
        ("transport", 30),
        ("woodworking", 30),
        ("zenotravel", 20),
    ]
    

    paths = []
    for domain in domains:
        temp_domain = domain[0]
        num_instances = domain[1]
        for instance in range(num_instances):
            if (instance<9):
                str_instance = "0"+str(instance+1)
            else:
                str_instance = str(instance+1)
            paths.append((main_path+"/"+temp_domain+"/task"+str_instance+".pddl.csv", ready_data_path+"/"+temp_domain+"_"+str_instance+"_x_.pt"))

    for path_tuple in paths:
        path = path_tuple[0]
        if os.path.isfile(os.getcwd()+path_tuple[1]) == False:
            continue
        column = ["level_1#", "level_1_F", "level_1_G", "node_max"]
        preprocessing_coumn = ["level_1#", "level_1_F", "level_1_G", "level_2#", "node_max"]
        df = pd.read_csv(os.getcwd()+path, skip_blank_lines=True, header=0, usecols=preprocessing_coumn)
        df = df.to_numpy()
        g_values = {}
        for i in range(len(df)):
            if (df[i][3] == 0):
                g_values[df[i][0]] = 0
            else:
                g_values[df[i][0]] = g_values[df[i][3]]+1
        df = pd.read_csv(os.getcwd()+path, skip_blank_lines=True, header=0, usecols=column)
        df = df.to_numpy()
        npbp = []
        pbp = []
        y = []
        max = None
        for i in tqdm.tqdm(range(len(df))):
            new_npbp = ((g_values[df[i][0]]) / (g_values[df[i][0]]+df[i][1]))*100
            npbp.append(new_npbp)
            if (max==None or new_npbp > max):
                max = new_npbp
            pbp.append(max)
            y.append(abs((df[i][0]/df[i][3]*100) - pbp[i]))
        x_df = pd.DataFrame(pbp)
        words = path.split("/")
        x_df.to_csv("pbp/new_"+words[-2]+"-"+words[-1])
        y_df = pd.DataFrame(y)
        y_df.to_csv("pbp/new_y_"+words[-2]+"-"+words[-1])
if __name__ == '__main__':
    main()


 
    
