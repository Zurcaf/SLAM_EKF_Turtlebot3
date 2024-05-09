import rosbag

# Caminho para o arquivo .bag
bag = rosbag.Bag('test.bag')

# Mostra as informações do arquivo .bag
print(bag)

# Mostra os tópicos do arquivo .bag
print(bag.get_type_and_topic_info())

# Mostra as mensagens do arquivo .bag
for topic, msg, t in bag.read_messages():
    print(f'{topic}: {msg}')
    
# Fecha o arquivo .bag
bag.close()
