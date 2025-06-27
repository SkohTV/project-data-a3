
document.addEventListener('DOMContentLoaded', function() {
    const astronaut = document.getElementById('astronaut');
    const astronautImg = astronaut.querySelector('img');
    
    if (!astronaut || !astronautImg) {



        return;
    }
    
    
    function zoomAstronaute(event) {
        
        const clientX = event.touches ? event.touches[0].clientX : event.clientX;
        const clientY = event.touches ? event.touches[0].clientY : event.clientY;
        
        
        const rect = astronautImg.getBoundingClientRect();
        
        
        const originX = ((clientX - rect.left) / rect.width) * 100;
        const originY = ((clientY - rect.top) / rect.height) * 100;
        
        
        astronautImg.style.transformOrigin = `${originX}% ${originY}%`;
        astronautImg.style.transition = 'transform 0.6s ease-in-out';
        
        
        astronautImg.style.transform = 'scale(2.5)';
        
        



        // dézoom
        setTimeout(() => {
            astronautImg.style.transform = 'scale(1)';
            
            
            setTimeout(() => {
                astronautImg.style.transformOrigin = '';
                astronautImg.style.transition = '';
            }, 600);
        }, 300);
        



    }
    
    




    
    astronaut.addEventListener('click', zoomAstronaute); // au clique de l'astronaute
    
    


    // tactile il faut tester car je ne suis pas sur que ca marche
    astronaut.addEventListener('touchstart', function(e) {
        e.preventDefault();
        zoomAstronaute(e);
    });
    
});
