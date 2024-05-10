import rosbag

try:
    # Abrir o arquivo ROS bag
    bag = rosbag.Bag('29-04-2024_17-43-03.bag')

    # Obtenha os tópicos do arquivo de bag
    topics = bag.get_type_and_topic_info().topics.keys()

    # Imprima os tópicos
    for topic in topics:
        print(topic)

    # Iterar pelas mensagens no bag
    # for topic, msg, t in bag.read_messages():
    #     # Processar as mensagens aqui
    #     print(f"Tópico: {topic}, Mensagem: {msg}, Timestamp: {t}")

    # Fechar o bag quando terminar
    bag.close()

except rosbag.ROSBagException as e:
    print(f"Erro ao abrir o arquivo ROS bag: {e}")
except FileNotFoundError:
    print("Arquivo não encontrado. Verifique o caminho ou nome do arquivo.")
