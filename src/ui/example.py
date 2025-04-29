from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class UIComponent:
    """UI 컴포넌트 기본 클래스"""
    id: str
    type: str
    props: Dict[str, Any]

class ExampleUI:
    """예제 UI 클래스"""
    
    def __init__(self):
        self.components: List[UIComponent] = []
    
    def add_component(self, component: UIComponent) -> None:
        """컴포넌트 추가"""
        self.components.append(component)
    
    def get_component(self, component_id: str) -> UIComponent:
        """컴포넌트 조회"""
        for component in self.components:
            if component.id == component_id:
                return component
        raise ValueError(f"Component not found: {component_id}")
    
    def render(self) -> Dict[str, Any]:
        """UI 렌더링"""
        return {
            "components": [
                {
                    "id": comp.id,
                    "type": comp.type,
                    "props": comp.props
                }
                for comp in self.components
            ]
        } 