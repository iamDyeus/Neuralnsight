document.addEventListener('DOMContentLoaded', function() {
// Get the toast element
  const toast = document.getElementById('toast-notification');

  // Function to show the toast at the specified position
  function showToast() {
    // Calculate the position of the toast relative to the viewport
    const toastRect = toast.getBoundingClientRect();
    const topPosition = window.scrollY + toastRect.top + 'px';
    const rightPosition = window.innerWidth - toastRect.right + 'px';

    // Set the position of the toast with transition effect
    toast.style.top = topPosition;
    toast.style.right = rightPosition;
    toast.style.transition = 'top 0.5s ease-out, right 0.5s ease-out';

    // Show the toast
    toast.style.display = 'block';

    // Increase the duration by 3 times and hide the toast after 15 seconds
    setTimeout(() => {
      hideToast();
    }, 8000); // Hide the toast after 8 seconds 

    // Optionally, you can add an event listener to hide the toast if it is clicked
    toast.addEventListener('click', () => {
      hideToast();
    });
  }

  // Function to hide the toast
  function hideToast() {
    // Hide the toast with transition effect
    toast.style.top = '-100px';
    toast.style.transition = 'top 0.5s ease-in';

    // Delay hiding the toast to allow the transition effect to complete
    setTimeout(() => {
      toast.style.display = 'none';
    }, 300);
  }

  // Call the showToast function to display the toast
  showToast();
});