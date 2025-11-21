from neo4j import GraphDatabase

uri = "neo4j://127.0.0.1:7687"
user = "neo4j"
password = "CRANE_HALL"

driver = GraphDatabase.driver(uri, auth=(user, password))

def inspect_nodes():
    with open('neo4j_inspection.txt', 'w') as f:
        # Check if any nodes exist
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN count(n) as count")
            f.write(f"Total nodes in DB: {result.single()['count']}\n")

        query = """
        MATCH (n)
        WHERE any(label in labels(n) WHERE label IN ['IfcDoor', 'IfcWindow'])
        RETURN n, labels(n)
        LIMIT 1
        """
        with driver.session() as session:
            result = session.run(query)
            record = result.single()
            if record:
                node = record['n']
                f.write(f"Labels: {record['labels(n)']}\n")
                f.write("Properties:\n")
                for key, value in node.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write("No IfcDoor or IfcWindow nodes found.\n")
                # Print one random node to see what it looks like
                result = session.run("MATCH (n) RETURN n, labels(n) LIMIT 1")
                record = result.single()
                if record:
                    f.write("Random node sample:\n")
                    f.write(f"Labels: {record['labels(n)']}\n")
                    f.write(f"{record['n'].items()}\n")

inspect_nodes()
driver.close()
