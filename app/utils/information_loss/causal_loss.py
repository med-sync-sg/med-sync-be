from app.utils.information_loss.information_loss import InformationLoss

class CausalLoss(InformationLoss):
    def __init__(self, name, description):
        super().__init__(name, description)
        
    def evaluate(self, original_text: str, simplified_text: str) -> float:
        """
        Evaluate information loss between original and simplified text.
        
        Args:
            original_text: Original medical text
            simplified_text: Simplified version of the text
            
        Returns:
            Loss value between 0 (no loss) and 1 (complete loss)
        """
        pass