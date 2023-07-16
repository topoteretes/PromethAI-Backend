from abc import ABC, abstractmethod
from typing import Optional

class AbstractAgent(ABC):

    def __init__(self, table_name=None, user_id: Optional[str] = "676", session_id: Optional[str] = None):
        self.table_name = table_name
        self.user_id = user_id
        self.session_id = session_id

    @abstractmethod
    def manage_resources(self, operation, resource_type, resource):
        """
        Perform the specified operation (e.g., add, update, delete) on the given resource type.
        """
        pass

    @abstractmethod
    def interact_with_database(self, operation, data):
        """
        Perform the specified operation (e.g., create, read, update, delete) on the database with the given data.
        """
        pass

    @abstractmethod
    def query(self, question):
        """Send a query to the agent and return the response."""
        pass

    @abstractmethod
    def terminate(self):
        """Perform any cleanup operations needed when the agent is no longer needed."""
        pass
